package semops

// Oracle is the expensive judge: given a predicate and a row's text, is the
// predicate true of it? This is the only thing that defines correctness; the
// cascade's whole job is to call it as few times as possible.
type Oracle interface {
	Judge(predicate, text string) bool
}

// Embedder turns text into a vector. Optional: when embeddings are already stored
// in the engine, an operator using a learned proxy never needs one.
type Embedder interface {
	Embed(texts []string) [][]float64
}

// Proxy is the cheap stand-in scored against the vector index. LogisticProxy is
// the concrete learned implementation; anything with these three methods works.
type Proxy interface {
	Fit(X [][]float64, y []bool)
	PredictProba(X [][]float64) []float64
	LinearParams() (weights []float64, bias float64, ok bool)
}

// NewProxy builds a fresh proxy of the same kind, for out-of-fold cross-fitting
// (each fold needs its own untrained model).
type NewProxy func() Proxy
