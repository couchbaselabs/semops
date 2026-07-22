package semops

// Row is one document flowing through an operator. Embedding may be nil when the
// engine left vectors in the store (the server-side-scoring path); operators that
// need it pull it on demand.
type Row struct {
	ID        string
	Text      string
	Embedding []float64
	Doc       map[string]any
}
