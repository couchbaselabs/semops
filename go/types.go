// Package semops is a Go port of the semantic-operators cascade library.
//
// The core is deliberately dependency-free, the
// heavy vector work is pushed into Couchbase, so the service only bands a bounded
// candidate set and makes HTTP calls. That makes this an I/O-bound service where
// Go's concurrency is a strict win and its lack of a numeric-library ecosystem
// does not matter.
package semops

// Band is which of the three cascade regions a row's proxy score falls into.
type Band int

const (
	// Reject: proxy is confident the predicate is FALSE. Dropped, no LLM call.
	Reject Band = iota
	// Escalate: proxy is uncertain. The oracle LLM is asked.
	Escalate
	// Accept: proxy is confident the predicate is TRUE. Kept, no LLM call.
	Accept
)

func (b Band) String() string {
	switch b {
	case Reject:
		return "reject"
	case Escalate:
		return "escalate"
	case Accept:
		return "accept"
	default:
		return "unknown"
	}
}

// Sample is one labeled calibration point: a proxy score and the oracle's verdict.
// Higher score means more likely TRUE, so the thresholds read naturally (accept
// above tauPlus, reject below tauMinus).
type Sample struct {
	Score float64
	Label bool
}
