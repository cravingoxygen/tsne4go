package tsne4go

import (
	"fmt"
	"math"
	"math/rand"
)

type TSne struct {
	perplexity float64
	dim        int
	epsilon    float64
	iter       int
	length     int
	probas     []float64
	Solution   [][]float64
	gains      [][]float64
	ystep      [][]float64
}

// return 0 mean unit standard deviation random number
func gaussRandom() float64 {
	u := 2*rand.Float64() - 1
	v := 2*rand.Float64() - 1
	r := u*u + v*v
	for r == 0 || r > 1 {
		u = 2*rand.Float64() - 1
		v = 2*rand.Float64() - 1
		r = u*u + v*v
	}
	c := math.Sqrt(-2 * math.Log(r) / r)
	return u * c
}

// return random normal number
func randn(mu, std float64) float64 {
	return mu + gaussRandom()*std
}

// returns 2d array filled with random numbers
func randn2d(n, d int) [][]float64 {
	res := make([][]float64, n)
	for i := range res {
		res[i] = make([]float64, d)
		for j := range res[i] {
			res[i][j] = randn(0.0, 1e-4)
		}
	}
	return res
}

// returns 2d array filled with 'val'
func fill2d(n, d int, val float64) [][]float64 {
	res := make([][]float64, n)
	for i := range res {
		res[i] = make([]float64, d)
		for j := range res[i] {
			res[i][j] = val
		}
	}
	return res
}

// compute pairwise distance in all vectors in X
func xtod(x Distancer) []float64 {
	length := x.Len()
	dists := make([]float64, length*length) // allocate contiguous array
	for i := 0; i < length-1; i++ {
		for j := i + 1; j < length; j++ {
			d := x.Distance(i, j)
			dists[i*length+j] = d
			dists[j*length+i] = d
		}
	}
	return dists
}

// "constants" for positive and negative infinity
var (
	inf    = math.Inf(1)
	negInf = math.Inf(-1)
)

// compute (p_{i|j} + p_{j|i})/(2n)
func d2p(D []float64, perplexity, tol float64) []float64 {
	Nf := math.Sqrt(float64(len(D))) // this better be an integer
	N := math.Floor(Nf)
	if N != Nf {
		panic("Should be a square")
	}
	length := int(N)
	Htarget := math.Log(perplexity)     // target entropy of distribution
	P := make([]float64, length*length) // temporary probability matrix
	prow := make([]float64, length)     // a temporary storage compartment
	for i := 0; i < length; i++ {
		betamin := negInf
		betamax := inf
		beta := 1.0 // initial value of precision
		done := false
		maxtries := 50
		// perform binary search to find a suitable precision beta
		// so that the entropy of the distribution is appropriate
		num := 0
		for !done {
			// compute entropy and kernel row with beta precision
			psum := 0.0
			for j := 0; j < length; j++ {
				if i != j { // we dont care about diagonals
					pj := math.Exp(-D[i*length+j] * beta)
					prow[j] = pj
					psum += pj
				} else {
					prow[j] = 0.0
				}
			}
			// normalize p and compute entropy
			Hhere := 0.0
			for j := 0; j < length; j++ {
				pj := prow[j] / psum
				prow[j] = pj
				if pj > 1e-7 {
					Hhere -= pj * math.Log(pj)
				}
			}
			// adjust beta based on result
			if Hhere > Htarget {
				// entropy was too high (distribution too diffuse)
				// so we need to increase the precision for more peaky distribution
				betamin = beta // move up the bounds
				if betamax == inf {
					beta = beta * 2
				} else {
					beta = (beta + betamax) / 2
				}
			} else {
				// converse case. make distrubtion less peaky
				betamax = beta
				if betamin == negInf {
					beta = beta / 2
				} else {
					beta = (beta + betamin) / 2
				}
			}
			// stopping conditions: too many tries or got a good precision
			num++
			if math.Abs(Hhere-Htarget) < tol {
				done = true
			}
			if num >= maxtries {
				done = true
			}
		}
		// copy over the final prow to P at row i
		for j := 0; j < length; j++ {
			P[i*length+j] = prow[j]
		}
	} // end loop over examples i
	// symmetrize P and normalize it to sum to 1 over all ij
	Pout := make([]float64, length*length)
	for i := 0; i < length; i++ {
		for j := 0; j < length; j++ {
			Pout[i*length+j] = math.Max((P[i*length+j]+P[j*length+i])/float64(length*2), 1e-100)
		}
	}
	return Pout
}

// helper function
func sign(x float64) int {
	if x > 0 {
		return 1
	} else if x < 0 {
		return -1
	}
	return 0
}

// this function takes a set of high-dimensional points
// and creates matrix P from them using gaussian kernel
func NewTSneDataRaw(x Distancer) *TSne {
	dists := xtod(x) // convert x to distances using gaussian kernel
	fmt.Printf("dists=%v\n", dists)
	tsne := &TSne{
		30,                   // perplexity
		1,                    // dim
		10,                   // epsilon
		0,                    // iters
		x.Len(),              //length
		d2p(dists, 30, 1e-4), // probas
		nil,                  // Solution
		nil,                  // gains
		nil,                  // ystep
	}
	tsne.initSolution() // refresh this
	return tsne
}

// this function takes a given distance matrix and creates
// matrix P from them.
// D is assumed to be provided as a list of lists, and should be symmetric
func NewTSneDataDist(dists [][]float64) *TSne {
	length := len(dists)
	// convert dists to an array
	vec := make([]float64, length*length)
	for i := 0; i < length-1; i++ {
		for j := i + 1; j < length; j++ {
			dist := dists[i][j]
			vec[i*length+j] = dist
			vec[j*length+i] = dist
		}
	}
	tsne := &TSne{
		30,                 // perplexity
		2,                  // dim
		10,                 // epsilon
		0,                  // iters
		length,             //length
		d2p(vec, 30, 1e-4), // probas
		nil,                // Solution
		nil,                // gains
		nil,                // ystep
	}
	tsne.initSolution() // refresh this
	return tsne
}

// (re)initializes the solution to random
func (tsne *TSne) initSolution() {
	// generate random solution to t-SNE
	tsne.Solution = randn2d(tsne.length, tsne.dim)  // the solution
	tsne.gains = fill2d(tsne.length, tsne.dim, 1.0) // step gains to accelerate progress in unchanging directions
	tsne.ystep = fill2d(tsne.length, tsne.dim, 0.0) // momentum accumulator
	tsne.iter = 0
}

// perform a single step of optimization to improve the embedding
func (tsne *TSne) Step() float64 {
	tsne.iter++
	length := tsne.length
	cost, grad := tsne.costGrad(tsne.Solution) // evaluate gradient
	// perform gradient step
	ymean := make([]float64, tsne.dim)
	for i := 0; i < length; i++ {
		for d := 0; d < tsne.dim; d++ {
			gid := grad[i][d]
			sid := tsne.ystep[i][d]
			gainid := tsne.gains[i][d]
			// compute gain update
			if sign(gid) == sign(sid) {
				tsne.gains[i][d] = gainid * 0.8
			} else {
				tsne.gains[i][d] = gainid + 0.2
			}
			// compute momentum step direction
			var momval float64
			if tsne.iter < 250 {
				momval = 0.5
			} else {
				momval = 0.8
			}
			newsid := momval*sid - tsne.epsilon*tsne.gains[i][d]*grad[i][d]
			tsne.ystep[i][d] = newsid // remember the step we took
			// step!
			tsne.Solution[i][d] += newsid
			ymean[d] += tsne.Solution[i][d] // accumulate mean so that we can center later
		}
	}
	// reproject Y to be zero mean
	for i := 0; i < length; i++ {
		for d := 0; d < tsne.dim; d++ {
			tsne.Solution[i][d] -= ymean[d] / float64(length)
		}
	}
	return cost
}

// return cost and gradient, given an arrangement
func (tsne *TSne) costGrad(Y [][]float64) (cost float64, grad [][]float64) {
	length := tsne.length
	dim := tsne.dim // dim of output space
	P := tsne.probas
	var pmul float64
	if tsne.iter < 100 { // trick that helps with local optima
		pmul = 4.0
	} else {
		pmul = 1.0
	}
	// compute current Q distribution, unnormalized first
	Qu := make([]float64, length*length)
	qsum := 0.0
	for i := 0; i < length-1; i++ {
		for j := i + 1; j < length; j++ {
			dsum := 0.0
			for d := 0; d < dim; d++ {
				dhere := Y[i][d] - Y[j][d]
				dsum += dhere * dhere
			}
			qu := 1.0 / (1.0 + dsum) // Student t-distribution
			Qu[i*length+j] = qu
			Qu[j*length+i] = qu
			qsum += 2 * qu
		}
	}
	// normalize Q distribution to sum to 1
	squareLength := length * length
	Q := make([]float64, squareLength)
	for q := range Q {
		Q[q] = math.Max(Qu[q]/qsum, 1e-100)
	}
	cost = 0.0
	grad = [][]float64{}
	for i := 0; i < length; i++ {
		gsum := make([]float64, dim) // init grad for point i
		for j := 0; j < length; j++ {
			cost += -P[i*length+j] * math.Log(Q[i*length+j]) // accumulate cost (the non-constant portion at least...)
			premult := 4 * (pmul*P[i*length+j] - Q[i*length+j]) * Qu[i*length+j]
			for d := 0; d < dim; d++ {
				gsum[d] += premult * (Y[i][d] - Y[j][d])
			}
		}
		grad = append(grad, gsum)
	}

	return cost, grad
}

// Normalize makes all values from the solution in the interval [0; 1]
func (tsne *TSne) NormalizeSolution() {
	mins := make([]float64, tsne.dim)
	maxs := make([]float64, tsne.dim)
	for i, pt := range tsne.Solution {
		for j, val := range pt {
			if i == 0 || val < mins[j] {
				mins[j] = val
			}
			if i == 0 || val > maxs[j] {
				maxs[j] = val
			}
		}
	}
	for i, pt := range tsne.Solution {
		for j, val := range pt {
			tsne.Solution[i][j] = (val - mins[j]) / (maxs[j] - mins[j])
		}
	}
}