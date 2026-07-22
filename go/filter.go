package semops

import (
	"math"
	"math/rand"
	"sync"
)

// FilterStats reports what the cascade did, for cost accounting and tests.
type FilterStats struct {
	NRows       int
	NSample     int
	TauMinus    float64
	TauPlus     float64
	NAccept     int
	NEscalate   int
	NReject     int
	OracleCalls int
	Collapsed   bool
}

// SemFilterInput carries a labeled candidate set for filtering. Embedding is
// required (the learned proxy needs it); Text feeds the oracle.
type SemFilterInput struct {
	Rows []Row
}

// SemFilterConfig sets targets and sampling.
type SemFilterConfig struct {
	Recall     float64
	Precision  float64
	Delta      float64
	SampleFrac float64
	MinSample  int
	MaxSample  int // 0 => uncapped
	Workers    int
	Seed       int64
	NewProxy   NewProxy // required: builds a fresh learned proxy
}

// DefaultFilterConfig matches the Python defaults for the learned-proxy path.
func DefaultFilterConfig() SemFilterConfig {
	return SemFilterConfig{
		Recall: 0.9, Precision: 0.9, Delta: 0.05,
		SampleFrac: 0.05, MinSample: 100, MaxSample: 0,
		Workers: 8, Seed: 0,
		NewProxy: func() Proxy { return NewLogisticProxy() },
	}
}

// SemFilter runs the three-band cascade over an in-memory candidate set using a
// learned proxy: sample and oracle-label, cross-fit to get honest thresholds,
// score every row with the final proxy, then band. Only the escalate band costs
// further oracle calls. Returns the kept rows and stats.
//
// This is the row-list path (the engine has already bounded the candidate set).
// The server-side index-served path is a separate entry point.
func SemFilter(rows []Row, predicate string, oracle Oracle, cfg SemFilterConfig) ([]Row, FilterStats) {
	n := len(rows)
	stats := FilterStats{NRows: n, TauMinus: NegInf, TauPlus: PosInf}
	if n == 0 {
		return nil, stats
	}
	var mu sync.Mutex
	call := func(r Row) bool {
		v := oracle.Judge(predicate, r.Text)
		mu.Lock()
		stats.OracleCalls++
		mu.Unlock()
		return v
	}

	// 1. sample + oracle-label
	idx := sampleIndices(n, cfg.SampleFrac, cfg.MinSample, cfg.MaxSample, cfg.Seed)
	stats.NSample = len(idx)
	sampleRows := make([]Row, len(idx))
	for i, j := range idx {
		sampleRows[i] = rows[j]
	}
	labels := pmap(sampleRows, cfg.Workers, call)

	X := make([][]float64, len(idx))
	for i, j := range idx {
		X[i] = rows[j].Embedding
	}

	// 2. cross-fit for honest calibration (see _oof_proba: a holdout split starves
	//    the Wilson bounds, so every labeled row is scored out-of-fold instead).
	oof := oofProba(cfg.NewProxy, X, labels, 5, cfg.Seed)
	calib := make([]Sample, len(oof))
	for i := range oof {
		calib[i] = Sample{Score: oof[i], Label: labels[i]}
	}
	th := Calibrate(calib, cfg.Recall, cfg.Precision, cfg.Delta)
	stats.TauMinus, stats.TauPlus, stats.Collapsed = th.TauMinus, th.TauPlus, th.Collapsed

	// 3. final proxy on the whole sample, score every row (probability space)
	proxy := cfg.NewProxy()
	proxy.Fit(X, labels)
	allX := make([][]float64, n)
	for i := range rows {
		allX[i] = rows[i].Embedding
	}
	scores := proxy.PredictProba(allX)

	// 4. band; sampled rows keep their oracle verdict, others are banded and the
	//    escalate band is sent to the oracle.
	sampled := make(map[int]bool, len(idx))
	labelOf := make(map[int]bool, len(idx))
	for i, j := range idx {
		sampled[j] = true
		labelOf[j] = labels[i]
	}
	keep := make([]bool, n)
	var escalateIdx []int
	for i := range rows {
		if sampled[i] {
			keep[i] = labelOf[i]
			continue
		}
		switch th.Band(scores[i]) {
		case Accept:
			keep[i] = true
			stats.NAccept++
		case Reject:
			stats.NReject++
		default:
			escalateIdx = append(escalateIdx, i)
		}
	}
	stats.NEscalate = len(escalateIdx)
	escRows := make([]Row, len(escalateIdx))
	for i, j := range escalateIdx {
		escRows[i] = rows[j]
	}
	verdicts := pmap(escRows, cfg.Workers, call)
	for i, j := range escalateIdx {
		keep[j] = verdicts[i]
	}

	out := make([]Row, 0, n)
	for i := range rows {
		if keep[i] {
			out = append(out, rows[i])
		}
	}
	return out, stats
}

// sampleIndices mirrors Python _sample_indices: target = clamp(max(min, ceil(frac*n)),
// maxSample, n), then a random sample of that size without replacement.
func sampleIndices(n int, frac float64, minSample, maxSample int, seed int64) []int {
	target := int(math.Ceil(frac * float64(n)))
	if minSample > target {
		target = minSample
	}
	if maxSample > 0 && target > maxSample {
		target = maxSample
	}
	if target > n {
		target = n
	}
	perm := rand.New(rand.NewSource(seed)).Perm(n)
	idx := perm[:target]
	return idx
}

// oofProba returns out-of-fold P(true) for every sample: each row is scored by a
// fold model that did not train on it, so all labels reach the calibrator while
// the thresholds stay honest. Mirrors Python _oof_proba.
func oofProba(newProxy NewProxy, X [][]float64, y []bool, folds int, seed int64) []float64 {
	n := len(X)
	out := make([]float64, n)
	for i := range out {
		out[i] = 0.5 // neutral prior for degenerate folds
	}
	order := rand.New(rand.NewSource(seed + 1)).Perm(n)
	for f := 0; f < folds; f++ {
		var testIdx []int
		for i := f; i < n; i += folds {
			testIdx = append(testIdx, order[i])
		}
		if len(testIdx) == 0 {
			continue
		}
		inTest := make(map[int]bool, len(testIdx))
		for _, i := range testIdx {
			inTest[i] = true
		}
		var trX [][]float64
		var trY []bool
		nPos := 0
		for i := 0; i < n; i++ {
			if inTest[i] {
				continue
			}
			trX = append(trX, X[i])
			trY = append(trY, y[i])
			if y[i] {
				nPos++
			}
		}
		if nPos == 0 || nPos == len(trY) { // degenerate fold: leave prior
			continue
		}
		m := newProxy()
		m.Fit(trX, trY)
		teX := make([][]float64, len(testIdx))
		for k, i := range testIdx {
			teX[k] = X[i]
		}
		p := m.PredictProba(teX)
		for k, i := range testIdx {
			out[i] = p[k]
		}
	}
	return out
}
