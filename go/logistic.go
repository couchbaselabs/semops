package semops

import "math"

// LogisticProxy is the learned cascade proxy: an L2-regularized logistic
// regression over embeddings, fit on the oracle-labeled calibration sample. It is
// the one piece with no Go-stdlib equivalent, so it is written from scratch here.
//
// Two properties are all the cascade needs, and both are guaranteed by
// construction:
//
//  1. PredictProba is monotonic in w·x + b, so the proxy ranks rows the same way a
//     vector index ranks them by dot product against w. That is what lets the
//     learned proxy be served by the index (the ann_above / DOT path).
//  2. LinearParams exposes (w, b) so the whole proxy can be pushed down as a single
//     VECTOR_DISTANCE(embedding, w, 'dot').
//
// The fit only ever runs on the calibration sample (a few hundred rows, once per
// query), so training speed is irrelevant in absolute terms; a plain full-batch
// gradient descent is more than adequate and keeps the core dependency-free.
type LogisticProxy struct {
	weights   []float64
	bias      float64
	fitted    bool
	constProb float64 // used when the sample is a single class

	// Hyperparameters. Defaults are set by NewLogisticProxy.
	LR    float64 // learning rate
	L2    float64 // L2 regularization strength
	Iters int     // gradient-descent iterations
}

// NewLogisticProxy returns a proxy with defaults tuned for unit-norm embeddings.
func NewLogisticProxy() *LogisticProxy {
	return &LogisticProxy{LR: 0.5, L2: 1e-4, Iters: 500}
}

func sigmoid(z float64) float64 {
	// Numerically stable: avoid Exp overflow for large-magnitude z.
	if z >= 0 {
		return 1.0 / (1.0 + math.Exp(-z))
	}
	e := math.Exp(z)
	return e / (1.0 + e)
}

// Fit trains the proxy on X (row-major embeddings) and boolean labels y.
//
// A degenerate single-class sample cannot support a decision boundary, so the
// proxy degrades to a constant probability (0 or 1) and LinearParams reports
// unusable, exactly as the Python SklearnLRProxy does when it sees one class.
func (p *LogisticProxy) Fit(X [][]float64, y []bool) {
	n := len(X)
	if n == 0 {
		p.fitted, p.constProb = false, 0.0
		return
	}
	pos := 0
	for _, v := range y {
		if v {
			pos++
		}
	}
	if pos == 0 || pos == n { // single class
		p.fitted = false
		if pos == n {
			p.constProb = 1.0
		} else {
			p.constProb = 0.0
		}
		return
	}

	dim := len(X[0])
	w := make([]float64, dim)
	b := 0.0
	gw := make([]float64, dim)

	for it := 0; it < p.Iters; it++ {
		for j := range gw {
			gw[j] = 0.0
		}
		gb := 0.0
		for i, xi := range X {
			z := b
			for j, v := range xi {
				z += w[j] * v
			}
			var target float64
			if y[i] {
				target = 1.0
			}
			err := sigmoid(z) - target
			for j, v := range xi {
				gw[j] += err * v
			}
			gb += err
		}
		invN := 1.0 / float64(n)
		for j := range w {
			// mean gradient + L2 penalty
			w[j] -= p.LR * (gw[j]*invN + p.L2*w[j])
		}
		b -= p.LR * gb * invN
	}

	p.weights, p.bias, p.fitted = w, b, true
}

// PredictProba returns P(label = true) for each row. When the fit was degenerate
// it returns the constant probability for every row.
func (p *LogisticProxy) PredictProba(X [][]float64) []float64 {
	out := make([]float64, len(X))
	if !p.fitted {
		for i := range out {
			out[i] = p.constProb
		}
		return out
	}
	for i, xi := range X {
		z := p.bias
		for j, v := range xi {
			z += p.weights[j] * v
		}
		out[i] = sigmoid(z)
	}
	return out
}

// LinearParams returns (weights, bias, true) so the proxy can be pushed to a
// vector index as a dot product. ok is false when the fit was degenerate and
// there is no usable linear model, mirroring the Python proxy returning None.
func (p *LogisticProxy) LinearParams() (weights []float64, bias float64, ok bool) {
	if !p.fitted {
		return nil, 0.0, false
	}
	return p.weights, p.bias, true
}
