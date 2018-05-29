// This is tsne4go, an implementation of the tSNE algorithm in Go. It is licensed under the MIT license.
// This library is strongly inspired from tsnejs, an equivalent library for javascript.
// Its goal is to reduce the dimension of data from a high-dimension space to a low-dimension (typically 2 ou 3) one.
// To learn more about it, you can download the scientific paper describing the tSNE algorithm.
package tsne4go

import (
	"fmt"
	"math"
)

const (
	perplexity float64 = 30
	epsilon    float64 = 5
	// NbDims is the number of dimensions in the target space, typically 2 or 3.
	NbDims      int = 2
	K               = 1.0
	alpha           = 1.0
	almost_zero     = 0.000001
)

var (
	sqrt_two = math.Sqrt(2.0)
)

// Point is a point's coordinates in the target space.
type Point [NbDims]float64

func copyPoints(src []Point) []Point {
	res := make([]Point, len(src))

	for k := range src {
		for i := 0; i < NbDims; i++ {
			res[k][i] = src[k][i]
		}
	}
	return res
}

// TSne is the main structure.
type TSne struct {
	iter         int
	length       int
	probas       []float64
	Solution     []Point // Solution is the list of points in the target space.
	gains        []Point
	ystep        []Point
	vars         []float64 //Variances for each point
	PrevSolution []Point
	prevDists    []float64 //x_i^t - x_i^{t+1} for all i

	// Meta gives meta-information about each point, if needed.
	// It is useful to associate, for instance, a label with each point.
	// The algorithm dosen't take this information into consideration.
	// It can be anything, even nil if the user has no need for it.
	Meta []interface{}
}

//Yeah, this is ugly and we will rework it, I promise
//Once my husband gives me a nice interface ;)

// New takes a set of Distancer instances
// and creates matrix P from them using gaussian kernel.
// Meta-information is provided here.
// It is under the programmer's responsibility :
// it can be nil if no meta information is needed, or anything else.
func New(x Distancer, xPrev Distancer, yPrev []Point, meta []interface{}) *TSne {
	dists := xtod(x) // convert x to distances using gaussian kernel
	length := x.Len()
	probas, vars := d2p(dists, 30, 1e-4)
	tsne := &TSne{
		iter:         0,                     // iters
		length:       length,                //length
		probas:       probas,                // probas
		vars:         vars,                  //variances
		gains:        fill2d(length, 1.0),   // gains
		ystep:        make([]Point, length), // ystep
		PrevSolution: yPrev,                 //Previous Solution
		Meta:         meta,
	}

	//Don't randomly re-initialize if previous points are known
	if yPrev != nil {
		tsne.Solution = randn2d(length) //copyPoints(yPrev)
	} else {
		tsne.Solution = randn2d(length)
	}

	//Initialize ||x^t - x^{t+1}||^2 since it will be constant for whole tsne
	if xPrev == nil || yPrev == nil {
		return tsne
	}

	if xPrev.Len() != x.Len() {
		panic("Number of high dim points must remain the same")
	}
	prevDists := make([]float64, xPrev.Len())
	for i := 0; i < xPrev.Len(); i++ {
		prevDists[i] = euclidSquared(x.Get(i), xPrev.Get(i))
	}
	tsne.prevDists = prevDists
	return tsne
}

// Step performs a single step of optimization to improve the embedding.
func (tsne *TSne) Step(verbose bool) float64 {
	tsne.iter++
	length := tsne.length
	cost, grad := tsne.costGrad(tsne.Solution, verbose) // evaluate gradient

	if math.IsNaN(cost) || grad == nil {
		panic("NaN")
	}

	// perform gradient step
	var ymean Point
	for i := 0; i < length; i++ {

		for d := 0; d < NbDims; d++ {
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
			momval := 0.8
			if tsne.iter < 250 {
				momval = 0.5
			}
			newsid := momval*sid - epsilon*tsne.gains[i][d]*grad[i][d]
			if verbose {
				//fmt.Printf("momval * sid - eps * gains * grad =\t %v * %v - %v * %v*%v = \t %v\n", momval, sid, epsilon, tsne.gains[i][d], grad[i][d], newsid)
			}
			tsne.ystep[i][d] = newsid // remember the step we took
			// step!
			tsne.Solution[i][d] += newsid
			ymean[d] += tsne.Solution[i][d] // accumulate mean so that we can center later
		}
	}

	// reproject Y to be zero mean
	for i := 0; i < length; i++ {
		for d := 0; d < NbDims; d++ {
			tsne.Solution[i][d] -= ymean[d] / float64(length)
		}
	}
	if verbose {
		fmt.Println("Solution", tsne.Solution[0])
	}
	return cost
}

// return cost and gradient, given an arrangement
func (tsne *TSne) costGrad(Y []Point, verbose bool) (cost float64, grad []Point) {
	length := tsne.length
	P := tsne.probas
	pmul := 1.0
	if tsne.iter < 100 { // trick that helps with local optima
		pmul = 4.0
	}
	// compute current Q distribution, unnormalized first
	Qu := make([]float64, length*length)
	qsum := 0.0
	for i := 0; i < length-1; i++ {
		for j := i + 1; j < length; j++ {
			dsum := 0.0
			for d := 0; d < NbDims; d++ {
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
	Q := make([]float64, length*length)
	for q := range Q {
		Q[q] = math.Max(Qu[q]/qsum, 1e-100)
	}
	cost = 0.0
	grad = make([]Point, length)
	for i := 0; i < length; i++ {
		for j := 0; j < length; j++ {
			idx := i*length + j
			cost += -P[idx] * math.Log(Q[idx]) // accumulate cost (the non-constant portion at least...)
			premult := 4 * (pmul*P[idx] - Q[idx]) * Qu[idx]
			for d := 0; d < NbDims; d++ {
				grad[i][d] += premult * (Y[i][d] - Y[j][d])
			}
		}
		if verbose && i == 1 {
			fmt.Println("tsne grad:", grad[i][0], grad[i][1])
		}
	}

	//Now we consider the second part of the cost function.
	//Would probably be more efficient to add it to the previous set of calculations, but for now we have it separate
	if tsne.PrevSolution != nil {

		//Find the edges of the rectangle containing all the Y points so that we can normalize
		mins, maxs := tsne.GetNormalizationBounds()
		var dists [NbDims]float64
		for d := range dists {
			dists[d] = (maxs[d] - mins[d]) //* (maxs[d] - mins[d])
		}

		for i := 0; i < length; i++ {

			//If no movement in the high dim space, then do nothing for now
			/*if math.Abs(tsne.prevDists[i]) < almost_zero {
				if verbose && i == 0 {
					fmt.Println("time cost 0")
				}
				continue
			} else {
			}*/

			//distance between y_i^t and y_i^{t+1}
			yPrevDists := (Y[i][0]-tsne.PrevSolution[i][0])*(Y[i][0]-tsne.PrevSolution[i][0]) + (Y[i][1]-tsne.PrevSolution[i][1])*(Y[i][1]-tsne.PrevSolution[i][1])

			//Same coeff for cost and the gradient in all dims
			coeff := 1.0                           //tsne.vars[i] * sqrt_two //sigma of qi = 1/sqrt(2), so sigma_pi / sigma_qi = sigma_pi * sqrt(2)
			coeff /= (K * (tsne.prevDists[i] + 1)) //Safe to divide

			timeCost := coeff*(yPrevDists) - 1 //* ((1000.0 - float64(tsne.iter)) / 1000.0)
			cost += math.Abs(timeCost)
			sign := 1.0
			if timeCost < 0 {
				sign *= -1
			}

			for d := 0; d < NbDims; d++ {
				gradi := pmul * 2.0 * coeff * (Y[i][d] - tsne.PrevSolution[i][d]) * sign
				gradi /= dists[d]
				grad[i][d] += gradi

				if math.IsNaN(grad[i][d]) {
					fmt.Println("NaN gradient: ", grad[i])
					fmt.Printf("2 * coeff * ( yPrev - Y )\t=  2 * %v * (%v - %v) \n", coeff, Y[i][d], tsne.PrevSolution[i][d])
					return cost, nil
				}
			}
			if verbose && i == 1 {
				fmt.Println("time grad", pmul*2.0*coeff*(Y[i][0]-tsne.PrevSolution[i][0])*sign, pmul*2.0*coeff*(Y[i][1]-tsne.PrevSolution[i][1])*sign)
			}
		}
	}
	return cost, grad
}

// NormalizeSolution makes all values from the solution in the interval [0; 1].
func (tsne *TSne) NormalizeSolution() []Point {
	mins, maxs := tsne.GetNormalizationBounds()

	res := make([]Point, len(tsne.Solution))

	for i, pt := range tsne.Solution {
		for j, val := range pt {
			res[i][j] = (val - mins[j]) / (maxs[j] - mins[j])
			tsne.Solution[i][j] = (val - mins[j]) / (maxs[j] - mins[j])
		}
	}
	return res
}

func (tsne *TSne) GetNormalizationBounds() (mins, maxs [NbDims]float64) {
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
	return mins, maxs
}
