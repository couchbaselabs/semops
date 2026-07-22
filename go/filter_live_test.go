//go:build live

// Live end-to-end test of the Go sem_filter against a running Couchbase cluster.
//
//	go test -tags live -run TestSemFilterLive -v ./...
//
// It needs a cluster_run cluster on :9499 with a `reviews` collection carrying
// stored `embedding` (384-d) and `label` fields, as produced by
//
//	examples/cb_ingest.py --dataset rotten --n 2000
//
// The oracle is the stored label (deterministic, free), so the measured F1 is the
// cascade's fidelity to the oracle, isolated from LLM noise. This exercises the
// full path: N1QL-over-REST scan, the learned logistic proxy, cross-fitted
// calibration, banding, and the escalate-band oracle calls.
package semops

import (
	"os"
	"testing"
)

const (
	liveQueryURL  = "http://localhost:9499"
	liveUser      = "Administrator"
	livePassword  = "asdasd"
	liveBucket    = "default"
	liveScope     = "_default"
	liveColl      = "reviews"
	livePredicate = "this is a negative or critical movie review"
)

type labelOracle struct{ byText map[string]bool }

func (o *labelOracle) Judge(predicate, text string) bool { return o.byText[text] }

func prf(truth, pred []bool) (p, r, f float64) {
	var tp, fp, fn int
	for i := range truth {
		switch {
		case pred[i] && truth[i]:
			tp++
		case pred[i] && !truth[i]:
			fp++
		case !pred[i] && truth[i]:
			fn++
		}
	}
	if tp+fp > 0 {
		p = float64(tp) / float64(tp+fp)
	} else {
		p = 1
	}
	if tp+fn > 0 {
		r = float64(tp) / float64(tp+fn)
	} else {
		r = 1
	}
	if p+r > 0 {
		f = 2 * p * r / (p + r)
	}
	return
}

func TestSemFilterLive(t *testing.T) {
	if os.Getenv("SEMOPS_LIVE") == "" {
		t.Skip("set SEMOPS_LIVE=1 to run against the cluster")
	}
	eng := NewCouchbaseEngine(
		NewQueryCluster(liveQueryURL, liveUser, livePassword), liveBucket, liveScope)

	rows, err := eng.Scan(liveColl, 0, true)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(rows) == 0 {
		t.Fatal("no rows: ingest the reviews collection first")
	}
	t.Logf("scanned %d rows, vectors pulled from cluster: %d", len(rows), eng.VectorsPulled)

	byText := make(map[string]bool, len(rows))
	truth := make([]bool, len(rows))
	filterRows := make([]Row, len(rows))
	pos := 0
	for i, r := range rows {
		byText[r.Text] = r.Label
		truth[i] = r.Label
		filterRows[i] = r.Row
		if r.Label {
			pos++
		}
	}
	t.Logf("positive (negative-review) rate: %.1f%%", 100*float64(pos)/float64(len(rows)))

	oracle := &labelOracle{byText: byText}
	cfg := DefaultFilterConfig()
	kept, stats := SemFilter(filterRows, livePredicate, oracle, cfg)

	// Reconstruct the boolean prediction aligned to truth, for P/R/F1.
	keptIDs := make(map[string]bool, len(kept))
	for _, r := range kept {
		keptIDs[r.ID] = true
	}
	pred := make([]bool, len(rows))
	for i, r := range rows {
		pred[i] = keptIDs[r.ID]
	}
	p, r, f := prf(truth, pred)
	savings := float64(stats.NRows) / float64(max(stats.OracleCalls, 1))

	t.Logf("cascade: P=%.3f R=%.3f F1=%.3f", p, r, f)
	t.Logf("oracle calls %d / %d rows  savings %.2fx", stats.OracleCalls, stats.NRows, savings)
	t.Logf("bands: accept=%d escalate=%d reject=%d  tau=(%.3f, %.3f)",
		stats.NAccept, stats.NEscalate, stats.NReject, stats.TauMinus, stats.TauPlus)

	// Fidelity to the oracle should be high (Python measured F1 ~0.987 here).
	if f < 0.95 {
		t.Errorf("F1 %.3f below 0.95; cascade is losing rows the oracle would keep", f)
	}
	// The whole point is spending fewer than one oracle call per row.
	if stats.OracleCalls >= stats.NRows {
		t.Errorf("no savings: %d calls for %d rows", stats.OracleCalls, stats.NRows)
	}
}
