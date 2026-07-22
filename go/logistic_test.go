package semops

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

// The logistic proxy cannot be pinned to sklearn's exact weights (different
// solver and regularization), so these are property tests: they assert the two
// things the cascade actually relies on, plus that it learns a known signal.

func TestLogisticSeparable(t *testing.T) {
	// A cleanly separable 2-D problem: class depends on x0 > x1.
	rng := rand.New(rand.NewSource(1))
	var X [][]float64
	var y []bool
	for i := 0; i < 400; i++ {
		a, b := rng.NormFloat64(), rng.NormFloat64()
		X = append(X, []float64{a, b})
		y = append(y, a > b)
	}
	p := NewLogisticProxy()
	p.Fit(X, y)

	w, _, ok := p.LinearParams()
	if !ok {
		t.Fatal("expected a usable linear model on separable data")
	}
	// The learned direction should point along (x0 - x1): w0 > 0, w1 < 0.
	if w[0] <= 0 || w[1] >= 0 {
		t.Errorf("weights do not recover the x0>x1 boundary: %v", w)
	}

	// Training accuracy should be high on a separable problem.
	probs := p.PredictProba(X)
	correct := 0
	for i := range X {
		if (probs[i] >= 0.5) == y[i] {
			correct++
		}
	}
	if acc := float64(correct) / float64(len(X)); acc < 0.95 {
		t.Errorf("training accuracy %.3f too low on separable data", acc)
	}
}

func TestLogisticMonotoneInDotProduct(t *testing.T) {
	// The load-bearing property: PredictProba must be monotone in w·x + b, which
	// is what lets a vector index serve the proxy by ranking on the dot product.
	rng := rand.New(rand.NewSource(2))
	var X [][]float64
	var y []bool
	for i := 0; i < 300; i++ {
		x := make([]float64, 8)
		s := 0.0
		for j := range x {
			x[j] = rng.NormFloat64()
			s += x[j]
		}
		X = append(X, x)
		y = append(y, s > 0)
	}
	p := NewLogisticProxy()
	p.Fit(X, y)
	w, b, ok := p.LinearParams()
	if !ok {
		t.Fatal("expected usable model")
	}

	probs := p.PredictProba(X)
	type pair struct {
		dot  float64
		prob float64
	}
	pairs := make([]pair, len(X))
	for i, xi := range X {
		d := b
		for j, v := range xi {
			d += w[j] * v
		}
		pairs[i] = pair{d, probs[i]}
	}
	sort.Slice(pairs, func(a, b int) bool { return pairs[a].dot < pairs[b].dot })
	for i := 1; i < len(pairs); i++ {
		if pairs[i].prob < pairs[i-1].prob-1e-9 {
			t.Fatalf("PredictProba not monotone in dot product at i=%d: %.6f < %.6f",
				i, pairs[i].prob, pairs[i-1].prob)
		}
	}
}

func TestLogisticSingleClass(t *testing.T) {
	X := [][]float64{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}
	p := NewLogisticProxy()

	p.Fit(X, []bool{true, true, true})
	if _, _, ok := p.LinearParams(); ok {
		t.Error("single-class fit should not yield a usable linear model")
	}
	for _, v := range p.PredictProba(X) {
		if v != 1.0 {
			t.Errorf("all-positive sample should predict 1.0, got %v", v)
		}
	}

	p.Fit(X, []bool{false, false, false})
	for _, v := range p.PredictProba(X) {
		if v != 0.0 {
			t.Errorf("all-negative sample should predict 0.0, got %v", v)
		}
	}
}

// TestLogisticRecoversDirection checks the proxy recovers a known ground-truth
// weight vector well enough to rank by it: AUC against the true scores near 1.
func TestLogisticRecoversDirection(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	trueW := []float64{1.5, -2.0, 0.0, 0.7}
	var X [][]float64
	var y []bool
	for i := 0; i < 500; i++ {
		x := make([]float64, len(trueW))
		z := 0.0
		for j := range x {
			x[j] = rng.NormFloat64()
			z += trueW[j] * x[j]
		}
		X = append(X, x)
		y = append(y, sigmoid(z) > rng.Float64())
	}
	p := NewLogisticProxy()
	p.Fit(X, y)
	probs := p.PredictProba(X)

	if auc := rankAUC(probs, y); auc < 0.85 {
		t.Errorf("AUC %.3f too low; proxy failed to recover the signal", auc)
	}
}

// rankAUC computes the area under the ROC curve via the rank-sum (Mann-Whitney)
// identity: AUC = (sum of ranks of positives - nPos*(nPos+1)/2) / (nPos*nNeg).
func rankAUC(scores []float64, y []bool) float64 {
	type sl struct {
		s float64
		y bool
	}
	arr := make([]sl, len(scores))
	for i := range scores {
		arr[i] = sl{scores[i], y[i]}
	}
	sort.Slice(arr, func(a, b int) bool { return arr[a].s < arr[b].s })
	// average ranks for ties
	ranks := make([]float64, len(arr))
	i := 0
	for i < len(arr) {
		j := i
		for j < len(arr) && arr[j].s == arr[i].s {
			j++
		}
		avg := float64(i+j+1) / 2.0 // ranks are 1-based; average of i+1..j
		for k := i; k < j; k++ {
			ranks[k] = avg
		}
		i = j
	}
	var sumPos float64
	var nPos, nNeg int
	for k := range arr {
		if arr[k].y {
			sumPos += ranks[k]
			nPos++
		} else {
			nNeg++
		}
	}
	if nPos == 0 || nNeg == 0 {
		return 0.5
	}
	return (sumPos - float64(nPos*(nPos+1))/2.0) / float64(nPos*nNeg)
}

var _ = math.Inf
