package semops

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// QueryCluster is a minimal N1QL driver over the Query REST API (/query/service),
// the Go counterpart of the Python HttpQueryCluster. It works against cluster_run
// dev clusters on non-standard ports where the SDK is awkward, and has no
// dependencies beyond net/http.
type QueryCluster struct {
	url    string
	auth   string
	client *http.Client
}

// NewQueryCluster targets a query node, e.g. "http://localhost:9499".
func NewQueryCluster(queryURL, username, password string) *QueryCluster {
	return &QueryCluster{
		url:    queryURL,
		auth:   "Basic " + base64.StdEncoding.EncodeToString([]byte(username+":"+password)),
		client: &http.Client{Timeout: 180 * time.Second},
	}
}

// Query runs a statement with named parameters ($name) and returns the result
// rows. Transient KV errors (bulk-get i/o timeouts, code 12008, which Couchbase
// flags retry:true) are retried with backoff; genuine errors like a syntax
// mistake fail immediately.
func (q *QueryCluster) Query(statement string, params map[string]any) ([]map[string]any, error) {
	payload := map[string]any{"statement": statement}
	for k, v := range params {
		if k[0] != '$' {
			k = "$" + k
		}
		payload[k] = v
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	const maxAttempts = 5
	backoff := 300 * time.Millisecond
	var lastErr error
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if attempt > 0 {
			time.Sleep(backoff)
			backoff *= 2
		}
		rows, err := q.do(body)
		if err == nil {
			return rows, nil
		}
		lastErr = err
		if !isRetryable(err) {
			return nil, err
		}
	}
	return nil, fmt.Errorf("after %d attempts: %w", maxAttempts, lastErr)
}

func (q *QueryCluster) do(body []byte) ([]map[string]any, error) {
	req, err := http.NewRequest(http.MethodPost, q.url+"/query/service", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", q.auth)

	resp, err := q.client.Do(req)
	if err != nil {
		return nil, err // network-level: retryable
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var out struct {
		Status  string           `json:"status"`
		Results []map[string]any `json:"results"`
		Errors  any              `json:"errors"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("decoding N1QL response: %w", err)
	}
	if out.Status != "success" {
		return nil, fmt.Errorf("N1QL error: %v", out.Errors)
	}
	return out.Results, nil
}

// isRetryable reports whether an error is a transient cluster condition worth
// retrying rather than a permanent failure.
func isRetryable(err error) bool {
	s := err.Error()
	return strings.Contains(s, "12008") || // KV bulk-get timeout, flagged retry:true
		strings.Contains(s, "i/o timeout") ||
		strings.Contains(s, "retry:true") ||
		strings.Contains(s, "connection refused") ||
		strings.Contains(s, "EOF")
}

// CouchbaseEngine is the native, first-priority engine. It pushes ANN retrieval
// and server-side scoring into the cluster; the service works on the bounded set
// that comes back.
type CouchbaseEngine struct {
	cluster     *QueryCluster
	bucket      string
	scope       string
	vectorField string
	textField   string
	metric      string
	nProbes     int
	rerank      bool
	keyChunk    int
	pageSize    int // rows per scan page, to keep each KV fetch small

	VectorsPulled int // count of embeddings shipped out of the cluster (for demos)
}

// NewCouchbaseEngine wires an engine to a query cluster.
func NewCouchbaseEngine(cluster *QueryCluster, bucket, scope string) *CouchbaseEngine {
	return &CouchbaseEngine{
		cluster:     cluster,
		bucket:      bucket,
		scope:       scope,
		vectorField: "embedding",
		textField:   "text",
		metric:      "cosine",
		nProbes:     8,
		rerank:      true,
		keyChunk:    512,
		pageSize:    500,
	}
}

func (e *CouchbaseEngine) keyspace(source string) string {
	return fmt.Sprintf("`%s`.`%s`.`%s`", e.bucket, e.scope, source)
}

func asFloat(v any) float64 {
	if f, ok := v.(float64); ok {
		return f
	}
	return 0
}

func asFloatSlice(v any) []float64 {
	arr, ok := v.([]any)
	if !ok {
		return nil
	}
	out := make([]float64, len(arr))
	for i, x := range arr {
		out[i] = asFloat(x)
	}
	return out
}

// ScanRow is what Scan returns: the fields the cascade needs plus the stored
// embedding and label (when present).
type ScanRow struct {
	Row
	Label bool
}

// Scan reads a collection. withVectors=false omits the embedding so vectors stay
// in the cluster; the label is always projected for label-oracle evaluation.
//
// It pages with LIMIT/OFFSET so no single request has to fetch the whole
// collection's documents at once: pulling thousands of 384-d embeddings in one
// KV fetch times the client out (error 12008), the same limit FetchVectors
// chunks around.
func (e *CouchbaseEngine) Scan(source string, limit int, withVectors bool) ([]ScanRow, error) {
	proj := fmt.Sprintf("META(d).id AS _id, d.`%s` AS _text, d.label AS _label", e.textField)
	if withVectors {
		proj += fmt.Sprintf(", d.`%s` AS _vec", e.vectorField)
	}
	var rows []ScanRow
	for offset := 0; ; offset += e.pageSize {
		page := e.pageSize
		if limit > 0 && offset+page > limit {
			page = limit - offset
		}
		if page <= 0 {
			break
		}
		stmt := fmt.Sprintf("SELECT %s FROM %s d LIMIT %d OFFSET %d",
			proj, e.keyspace(source), page, offset)
		res, err := e.cluster.Query(stmt, nil)
		if err != nil {
			return nil, err
		}
		for _, r := range res {
			id, _ := r["_id"].(string)
			text, _ := r["_text"].(string)
			label, _ := r["_label"].(bool)
			sr := ScanRow{Row: Row{ID: id, Text: text}, Label: label}
			if withVectors {
				sr.Embedding = asFloatSlice(r["_vec"])
				if sr.Embedding != nil {
					e.VectorsPulled++
				}
			}
			rows = append(rows, sr)
		}
		if len(res) < page { // last page
			break
		}
	}
	return rows, nil
}

// FetchVectors pulls stored embeddings for specific keys, chunked to stay under
// the KV bulk-get limit (see gsi_notes.md: 25k keys in one USE KEYS times out
// with error 12008).
func (e *CouchbaseEngine) FetchVectors(source string, keys []string) (map[string][]float64, error) {
	out := make(map[string][]float64, len(keys))
	stmt := fmt.Sprintf("SELECT META(d).id AS id, d.`%s` AS v FROM %s d USE KEYS $keys",
		e.vectorField, e.keyspace(source))
	for i := 0; i < len(keys); i += e.keyChunk {
		end := i + e.keyChunk
		if end > len(keys) {
			end = len(keys)
		}
		res, err := e.cluster.Query(stmt, map[string]any{"keys": keys[i:end]})
		if err != nil {
			return nil, err
		}
		for _, r := range res {
			id, _ := r["id"].(string)
			if v := asFloatSlice(r["v"]); v != nil {
				out[id] = v
				e.VectorsPulled++
			}
		}
	}
	return out, nil
}

// Count returns the number of documents in a collection.
func (e *CouchbaseEngine) Count(source string) (int, error) {
	res, err := e.cluster.Query(
		fmt.Sprintf("SELECT COUNT(*) AS c FROM %s", e.keyspace(source)), nil)
	if err != nil {
		return 0, err
	}
	if len(res) == 0 {
		return 0, nil
	}
	return int(asFloat(res[0]["c"])), nil
}
