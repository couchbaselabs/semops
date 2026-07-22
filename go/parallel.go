package semops

import "sync"

// pmap applies fn to every item across a bounded pool of goroutines, preserving
// order. This is the Go replacement for the Python thread pool: no GIL, so the
// per-item work runs in genuine parallel, and the escalate-band oracle calls
// (I/O-bound) overlap.
func pmap[T, R any](items []T, workers int, fn func(T) R) []R {
	out := make([]R, len(items))
	if workers < 1 {
		workers = 1
	}
	if workers > len(items) {
		workers = len(items)
	}
	if workers <= 1 {
		for i, it := range items {
			out[i] = fn(it)
		}
		return out
	}
	var wg sync.WaitGroup
	ch := make(chan int)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range ch {
				out[i] = fn(items[i])
			}
		}()
	}
	for i := range items {
		ch <- i
	}
	close(ch)
	wg.Wait()
	return out
}
