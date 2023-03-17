package main

import (
	"errors"
	"fmt"
	"plaintext/src/mnist"
	"plaintext/src/model"
	"sort"
)

// Paper source: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/04/CryptonetsTechReport.pdf

const (
	IMAGE_WIDTH           = 28
	IMAGE_CROP            = 13
	WINDOW_WIDTH          = 5
	STRIDE                = 2
	OUTPUT_CHANNELS       = 5
	FINAL_OUTPUT_CHANNELS = 10
	FULLY_CONNECTED_WIDTH = 100
	CONVOLUTION_SIZE      = IMAGE_CROP * IMAGE_CROP // 169
)

func pad_image(input []float64) []float64 {
	// Pad the upper and left edge with 0s
	padded_width := IMAGE_WIDTH + 2
	output := make([]float64, padded_width*padded_width)
	input_index := 0
	for i := padded_width*2; i < padded_width*padded_width; i++ {
		if i%padded_width < 2 {
			continue
		}
		output[i] = input[input_index]
		input_index++
	}
	
	for i := 0; i < padded_width; i++ {
		for j := 0; j < padded_width; j++ {
			if output[i*padded_width+j] > 0 {
				fmt.Print(1, " ")
			} else {
				fmt.Print(0, " ")
			}
		} 
		fmt.Println()
	}

	return output
}

func convolution(input []float64, kernel []float64) []float64 {
	output := make([]float64, CONVOLUTION_SIZE)
	for y := 0; y < IMAGE_CROP; y++ {
		for x := 0; x < IMAGE_CROP; x++ {
			val := 0.
			for wy := 0; wy < WINDOW_WIDTH; wy++ {
				for wx := 0; wx < WINDOW_WIDTH; wx++ {
					kernel_pos := wx + wy*WINDOW_WIDTH
					val += kernel[kernel_pos] * input[(x*STRIDE+wx)+(y*STRIDE+wy)*IMAGE_CROP]
				}
			}
			output[x+y*IMAGE_CROP] = val
		}
	}
	return output
}

func convolution_layer(input []float64, weights []float64) []float64 {
	output := make([]float64, OUTPUT_CHANNELS*CONVOLUTION_SIZE)
	kernel_size := WINDOW_WIDTH * WINDOW_WIDTH
	kernel := make([]float64, kernel_size)
	for o := 0; o < OUTPUT_CHANNELS; o++ {
		// make kernel from weights
		for i := 0; i < kernel_size; i++ {
			kernel[i] = weights[o*kernel_size+i]
		}
		conv := convolution(input, kernel)
		for i := 0; i < CONVOLUTION_SIZE; i++ {
			output[o*CONVOLUTION_SIZE+i] = conv[i]
		}
	}
	return output
}

func square(input []float64) []float64 {
	// This layer squares the value at each input node.
	var output []float64
	for _, e := range input {
		output = append(output, e*e)
	}
	return output
}

func transpose(input []float64, input_shape int, output_maps int) []float64 {
	output := make([]float64, len(input))
	for i := 0; i < input_shape; i++ {
		for j := 0; j < output_maps; j++ {
			output[i+input_shape*j] = input[output_maps*i+j]
		}
	}
	return output
}

func dot_product(a []float64, b []float64) (float64, error) {
	if len(a) != len(b) {
		return 0., errors.New("Input vectors should be same length.")
	}
	sum := 0.
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum, nil
}

func pool_layer(input []float64, weights []float64) ([]float64, error) {
	// The other change that we make to the network is just for an increase
	// in efficiency. Since layers 3 through 6 are all linear, they can all
	// be viewed as matrix multiplication and composed into a single linear
	// layer corresponding to a matrix of dimension 100 by 5 · 13 · 13 = 865.
	// Thus, our final network for making predictions is only 5 layers deep.

	// 100 = (1 / 5 x 13 x 13) * (5 x 13 x 13 / 100)
	output := make([]float64, FULLY_CONNECTED_WIDTH)
	pool_size := CONVOLUTION_SIZE * OUTPUT_CHANNELS
	for i := 0; i < FULLY_CONNECTED_WIDTH; i++ {
		pool := make([]float64, pool_size)
		for j := 0; j < pool_size; j++ {
			pool[j] = weights[pool_size*i+j]
		}
		out, err := dot_product(input, pool)
		if err != nil {
			return output, err
		}
		output[i] = out
	}
	return output, nil
}

func output_layer(input []float64, weights []float64) ([]float64, error) {
	// 1 = (10 / 100) * (100 / 10)
	output := make([]float64, FINAL_OUTPUT_CHANNELS)
	for i := 0; i < FINAL_OUTPUT_CHANNELS; i++ {
		pool := make([]float64, FULLY_CONNECTED_WIDTH)
		for j := 0; j < FULLY_CONNECTED_WIDTH; j++ {
			pool[j] = weights[FULLY_CONNECTED_WIDTH*i+j]
		}
		out, err := dot_product(input, pool)
		if err != nil {
			return output, err
		}
		output[i] = out
	}
	return output, nil
}

func cryptonets(input []float64) ([]float64, error) {
	// Inference layers (see Table 1 of `Paper source`)

	// 1. Convolution layer
	// Weighted sums layer with windows of size 5×5, stride size of 2. From
	// each window, 5 different maps are computed and a padding is added to
	// the upper side and left side of each image.
	conv := convolution_layer(input, model.Weights_0)

	// 2. Square layer
	// Squares each of the 835 outputs of the convolution layer
	sq_1 := square(conv)

	// Transpose Weights_1
	weights_1 := transpose(model.Weights_1, OUTPUT_CHANNELS*CONVOLUTION_SIZE, FULLY_CONNECTED_WIDTH)

	// 3. Pool layer
	// Weighted sum layer that generates 100 outputs from the 835 outputs of
	// 1st the square layer
	pool, err := pool_layer(sq_1, weights_1)
	if err != nil {
		return pool, err
	}

	// 4. Square layer
	// Squares each of the 100 outputs of the pool layer
	sq_2 := square(pool)

	// 5. Output layer
	// Weighted sum that generates 10 outputs (corresponding to the 10 digits)
	// from the 100 outputs of the 2nd square layer
	output, err_2 := output_layer(sq_2, model.Weights_3)
	if err_2 != nil {
		return output, err_2
	}

	return output, nil
}

func main() {
	// Load MNIST datasets
	dataSet, err := mnist.ReadTestSet("./mnist")
	if err != nil {
		fmt.Println(err)
		return
	}
	digit_image := dataSet.Data[9]

	// Format input
	input := make([]float64, IMAGE_WIDTH*IMAGE_WIDTH)
	for i := 0; i < IMAGE_WIDTH; i++ {
		for j := 0; j < IMAGE_WIDTH; j++ {
			input[i*IMAGE_WIDTH+j] = float64(digit_image.Image[i][j])
			if digit_image.Image[i][j] > 0 {
				fmt.Print(1, " ")
			} else {
				fmt.Print(0, " ")
			}
		}
		fmt.Println()
	}

	for i := 0; i < IMAGE_WIDTH; i++ {
		for j := 0; j < IMAGE_WIDTH; j++ {
			input[i*IMAGE_WIDTH+j] = float64(digit_image.Image[i][j])
		}
	}

	// The upper and left side of each image is padded
	padded_image := pad_image(input)
	
	// Run cryptonets
	res, err := cryptonets(padded_image)
	if err != nil {
		fmt.Println("error:", err)
	} else {
		// Print result
		pred := 0
		pred_r := 0.
		for i, r := range res {
			if r < pred_r {
				pred = i
				pred_r = r
			}
			fmt.Println("i: ", i, ", r:", r)
		}
		fmt.Println("digit:", digit_image.Digit)
		fmt.Println("Pred:", pred)

		sort.Float64s(res)
		fmt.Println("sorted", res)
	}
}
