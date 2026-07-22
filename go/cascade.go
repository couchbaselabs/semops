package semops

import (
	"math"
	"sort"
)

// NegInf and PosInf are the "no auto-decision" thresholds: tauMinus = -inf means
// reject nothing, tauPlus = +inf means accept nothing. Both push rows to the oracle.
var (
	NegInf = math.Inf(-1)
	PosInf = math.Inf(1)
)

// Thresholds are the learned cascade boundaries. See Calibrate.
type Thresholds struct {
	TauMinus  float64
	TauPlus   float64
	Collapsed bool // thresholds crossed: a single decision boundary, no escalate band
}

// Band classifies a proxy score against the learned thresholds.
func (t Thresholds) Band(score float64) Band {
	if score >= t.TauPlus {
		return Accept
	}
	if score <= t.TauMinus {
		return Reject
	}
	return Escalate
}

// normInvCDF is the inverse of the standard normal CDF (the probit function),
// via Wichura's algorithm AS 241 (PPND16). Full double precision, matching
// Python's statistics.NormalDist().inv_cdf bit-for-bit, so calibration thresholds
// are identical across the two implementations.
func normInvCDF(p float64) float64 {
	const (
		a0 = 3.3871328727963666080e0
		a1 = 1.3314166789178437745e+2
		a2 = 1.9715909503065514427e+3
		a3 = 1.3731693765509461125e+4
		a4 = 4.5921953931549871457e+4
		a5 = 6.7265770927008700853e+4
		a6 = 3.3430575583588128105e+4
		a7 = 2.5090809287301226727e+3
		b1 = 4.2313330701600911252e+1
		b2 = 6.8718700749205790830e+2
		b3 = 5.3941960214247511077e+3
		b4 = 2.1213794301586595867e+4
		b5 = 3.9307895800092710610e+4
		b6 = 2.8729085735721942674e+4
		b7 = 5.2264952788528545610e+3
		c0 = 1.42343711074968357734e0
		c1 = 4.63033784615654529590e0
		c2 = 5.76949722146069140550e0
		c3 = 3.64784832476320460504e0
		c4 = 1.27045825245236838258e0
		c5 = 2.41780725177450611770e-1
		c6 = 2.27238449892691845833e-2
		c7 = 7.74545014278341407640e-4
		d1 = 2.05319162663775882187e0
		d2 = 1.67638483018380384940e0
		d3 = 6.89767334985100004550e-1
		d4 = 1.48103976427480074590e-1
		d5 = 1.51986665636164571966e-2
		d6 = 5.47593808499534494600e-4
		d7 = 1.05075007164441684324e-9
		e0 = 6.65790464350110377720e0
		e1 = 5.46378491116411436990e0
		e2 = 1.78482653991729133580e0
		e3 = 2.96560571828504891230e-1
		e4 = 2.65321895265761230930e-2
		e5 = 1.24266094738807843860e-3
		e6 = 2.71155556874348757815e-5
		e7 = 2.01033439929228813265e-7
		f1 = 5.99832206555887937690e-1
		f2 = 1.36929880922735805310e-1
		f3 = 1.48753612908506148525e-2
		f4 = 7.86869131145613259100e-4
		f5 = 1.84631831751005468180e-5
		f6 = 1.42151175831644588870e-7
		f7 = 2.04426310338993978564e-15
	)
	q := p - 0.5
	if math.Abs(q) <= 0.425 {
		r := 0.180625 - q*q
		return q * (((((((a7*r+a6)*r+a5)*r+a4)*r+a3)*r+a2)*r+a1)*r + a0) /
			(((((((b7*r+b6)*r+b5)*r+b4)*r+b3)*r+b2)*r+b1)*r + 1.0)
	}
	var r float64
	if q < 0 {
		r = p
	} else {
		r = 1.0 - p
	}
	if r <= 0 {
		return math.NaN()
	}
	r = math.Sqrt(-math.Log(r))
	var val float64
	if r <= 5.0 {
		r -= 1.6
		val = (((((((c7*r+c6)*r+c5)*r+c4)*r+c3)*r+c2)*r+c1)*r + c0) /
			(((((((d7*r+d6)*r+d5)*r+d4)*r+d3)*r+d2)*r+d1)*r + 1.0)
	} else {
		r -= 5.0
		val = (((((((e7*r+e6)*r+e5)*r+e4)*r+e3)*r+e2)*r+e1)*r + e0) /
			(((((((f7*r+f6)*r+f5)*r+f4)*r+f3)*r+f2)*r+f1)*r + 1.0)
	}
	if q < 0 {
		return -val
	}
	return val
}

// zForTwoSidedSplit returns z for a one-sided confidence of (1 - delta/2): each
// of the two guarantees (precision, recall) spends delta/2.
func zForTwoSidedSplit(delta float64) float64 {
	if !(delta > 0.0 && delta < 1.0) {
		panic("delta must be in (0, 1)")
	}
	return normInvCDF(1.0 - delta/2.0)
}

// WilsonLowerBound is the lower bound of the Wilson score interval for a
// proportion k/n at the given z. It is the conservative, finite-sample confidence
// bound the cascade uses so a guarantee is never emitted on a point estimate.
func WilsonLowerBound(k, n int, z float64) float64 {
	if n == 0 {
		return 0.0
	}
	phat := float64(k) / float64(n)
	z2 := z * z
	nf := float64(n)
	denom := 1.0 + z2/nf
	center := phat + z2/(2.0*nf)
	margin := z * math.Sqrt((phat*(1.0-phat)+z2/(4.0*nf))/nf)
	if v := (center - margin) / denom; v > 0.0 {
		return v
	}
	return 0.0
}

// Calibrate learns (tauMinus, tauPlus) from a labeled (score, label) sample.
//
// It degrades safely: if the proxy is too weak to certify a target, the
// corresponding threshold goes to infinity so those rows escalate to the oracle
// rather than being wrongly auto-labeled. No false guarantee is ever emitted.
//
// The logic mirrors the Python original exactly, including the collapse case
// where a strong-enough proxy decides everything and the escalate band vanishes.
func Calibrate(sample []Sample, recallTarget, precisionTarget, delta float64) Thresholds {
	n := len(sample)
	if n == 0 {
		return Thresholds{NegInf, PosInf, false} // everything escalates
	}
	if recallTarget < 0.0 || recallTarget > 1.0 || precisionTarget < 0.0 || precisionTarget > 1.0 {
		panic("targets must be in [0, 1]")
	}

	z := zForTwoSidedSplit(delta)

	// Distinct scores, ascending (mirrors Python's sorted(set(...))).
	seen := make(map[float64]struct{}, n)
	scores := make([]float64, 0, n)
	totalPos := 0
	for _, s := range sample {
		if _, ok := seen[s.Score]; !ok {
			seen[s.Score] = struct{}{}
			scores = append(scores, s.Score)
		}
		if s.Label {
			totalPos++
		}
	}
	sort.Float64s(scores)

	// tauPlus: the smallest t whose accept region {s >= t} has guaranteed
	// precision >= precisionTarget. Smallest passing t gives the largest region.
	tauPlus := PosInf
	for _, t := range scores { // ascending
		k, m := 0, 0
		for _, s := range sample {
			if s.Score >= t {
				m++
				if s.Label {
					k++
				}
			}
		}
		if m > 0 && WilsonLowerBound(k, m, z) >= precisionTarget {
			tauPlus = t
			break
		}
	}

	// tauMinus: the largest t whose kept set {s > t} has guaranteed recall >=
	// recallTarget. With no positives in the sample recall cannot be certified,
	// so we refuse to auto-reject (tauMinus stays -inf).
	tauMinus := NegInf
	if totalPos > 0 {
		for i := len(scores) - 1; i >= 0; i-- { // descending
			t := scores[i]
			posKept := 0
			for _, s := range sample {
				if s.Label && s.Score > t {
					posKept++
				}
			}
			if WilsonLowerBound(posKept, totalPos, z) >= recallTarget {
				tauMinus = t
				break
			}
		}
	}

	// If the bands cross (proxy strong enough to decide everything), collapse to a
	// single decision boundary with no escalate band.
	collapsed := false
	if !math.IsInf(tauMinus, -1) && !math.IsInf(tauPlus, 1) && tauMinus >= tauPlus {
		theta := (tauPlus + tauMinus) / 2.0
		tauPlus, tauMinus = theta, theta
		collapsed = true
	}

	return Thresholds{tauMinus, tauPlus, collapsed}
}

// EmpiricalPrecisionRecall reports precision and recall of a boolean prediction
// against the true labels. Used in evaluation and tests.
func EmpiricalPrecisionRecall(sample []Sample, predictedKeep []bool) (precision, recall float64) {
	tp, fp, fn := 0, 0, 0
	for i, s := range sample {
		p := predictedKeep[i]
		switch {
		case p && s.Label:
			tp++
		case p && !s.Label:
			fp++
		case !p && s.Label:
			fn++
		}
	}
	if tp+fp > 0 {
		precision = float64(tp) / float64(tp+fp)
	} else {
		precision = 1.0
	}
	if tp+fn > 0 {
		recall = float64(tp) / float64(tp+fn)
	} else {
		recall = 1.0
	}
	return precision, recall
}
