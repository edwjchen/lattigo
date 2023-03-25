package main

import (
	"fmt"
	"plain/src/mnist"
	"plain/src/model"
)

// Paper source: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/04/CryptonetsTechReport.pdf

const (
	SLOTS              = 8192            // number of slots within a plaintext
	IMAGE_WIDTH        = 28              // width of image
	PADDED_IMAGE_WIDTH = IMAGE_WIDTH + 1 // width of image after padding
	IMAGE_NORM         = 256             // normalization scale for images
	STRIDE             = 2               // stride of convolution window
	CONV_MAP           = 5               // number of convolution maps
	CONV_SIZE          = 13              // convolution width (and length)
	CONV_WINDOW_LEN    = 5               // size of convolution window
	CONVOLUTION_SIZE   = 845             // number of convolution layers
	POOL_LAYERS        = 100             // number of pooling layers
	FC_LAYERS          = 10              // number of fc layers
)

// Pad the right and bottom edges of an image with 0s
func pad_image(input [][]float64) [][]float64 {
	output := make([][]float64, PADDED_IMAGE_WIDTH)
	for i := 0; i < IMAGE_WIDTH; i++ {
		row := make([]float64, PADDED_IMAGE_WIDTH)
		for j := 0; j < IMAGE_WIDTH; j++ {
			row[j] = input[i][j]
		}
		row[IMAGE_WIDTH] = float64(0)
		output[i] = row
	}
	output[IMAGE_WIDTH] = make([]float64, PADDED_IMAGE_WIDTH)
	return output
}

// Get images from the mnist dataset, while padding and normalizing
// image values
//
// Returns `SLOTS` number of images and corresponding answer key
func get_images() ([][][]float64, []int) {
	dataSet, err := mnist.ReadTestSet("./mnist")
	if err != nil {
		fmt.Println(err)
		panic("MNIST Dataset error")
	}

	var images [][][]float64 = make([][][]float64, 0)
	var answers []int = make([]int, SLOTS)

	// pack a matrix of 8192 x 784
	// pad image size from 784 to 841
	for r := 0; r < SLOTS; r++ {
		image := dataSet.Data[r]
		answers[r] = image.Digit
		image_matrix := make([][]float64, IMAGE_WIDTH)
		for i := 0; i < IMAGE_WIDTH; i++ {
			image_row := make([]float64, IMAGE_WIDTH)
			for j := 0; j < IMAGE_WIDTH; j++ {
				image_row[j] = float64(image.Image[i][j])
			}
			image_matrix[i] = image_row
		}
		padded_image := pad_image(image_matrix)
		images = append(images, padded_image)
	}

	// normalize images
	for i := 0; i < SLOTS; i++ {
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			for k := 0; k < PADDED_IMAGE_WIDTH; k++ {
				images[i][j][k] /= IMAGE_NORM
			}
		}
	}

	return images, answers
}

// Prepare convolution layer model weights
func prepare_conv() ([][]float64, []float64) {
	return model.ConvKernels, model.ConvBias
}

// Prepare pool layer model weights
func prepare_pool_layer() ([][]float64, []float64) {
	pool_layers := make([][]float64, POOL_LAYERS)
	for i := 0; i < POOL_LAYERS; i++ {
		pool_layer := make([]float64, CONVOLUTION_SIZE)
		for j := 0; j < CONVOLUTION_SIZE; j++ {
			// transpose weights
			pool_layer[j] = model.Weights_1[i+j*POOL_LAYERS]
		}
		pool_layers[i] = pool_layer
	}
	return pool_layers, model.PoolBias
}

// Prepare fc layer model weights
func prepare_fc_layer() ([][]float64, []float64) {
	fc_layers := make([][]float64, FC_LAYERS)
	for i := 0; i < FC_LAYERS; i++ {
		fc_layer := make([]float64, POOL_LAYERS)
		for j := 0; j < POOL_LAYERS; j++ {
			fc_layer[j] = model.Weights_3[i*POOL_LAYERS+j]
		}
		fc_layers[i] = fc_layer
	}
	return fc_layers, model.FcBias
}

// Perform the convolution layer
//
// Weighted sums layer with windows of size 5×5, stride size of 2. From
// each window, 5 different maps are computed.
func convolution_layer(image [][]float64) [][][]float64 {
	conv_kernels, conv_bias := prepare_conv()

	outputs := make([][][]float64, CONV_MAP)
	for o := 0; o < CONV_MAP; o++ {
		row_output := make([][]float64, CONV_SIZE)
		windows := make([][]float64, CONV_SIZE*CONV_SIZE)

		for stride_y := 0; stride_y < CONV_SIZE; stride_y++ {
			col_output := make([]float64, CONV_SIZE)
			for stride_x := 0; stride_x < CONV_SIZE; stride_x++ {
				// create window
				window := make([]float64, CONV_WINDOW_LEN*CONV_WINDOW_LEN)
				offset_x := stride_x * STRIDE
				offset_y := stride_y * STRIDE
				for i := 0; i < CONV_WINDOW_LEN; i++ {
					for j := 0; j < CONV_WINDOW_LEN; j++ {
						// padded image width is 29
						window[i*CONV_WINDOW_LEN+j] = image[j+offset_y][i+offset_x]
					}
				}
				windows[stride_y*CONV_SIZE+stride_x] = window

				kernel := conv_kernels[o]

				product := make([]float64, CONV_WINDOW_LEN*CONV_WINDOW_LEN)
				for i := 0; i < CONV_WINDOW_LEN*CONV_WINDOW_LEN; i++ {
					product[i] = window[i] * kernel[i]
				}

				// sum all products together
				sum := product[0]
				for i := 1; i < CONV_WINDOW_LEN*CONV_WINDOW_LEN; i++ {
					sum += product[i]
				}

				// add bias
				sum += conv_bias[o]

				col_output[stride_x] = sum

			}
			row_output[stride_y] = col_output
		}
		outputs[o] = row_output
	}

	return outputs
}

// Square all values
func square_layer_1(conv_layer [][][]float64) [][][]float64 {
	squared := make([][][]float64, CONV_MAP)
	for i := 0; i < CONV_MAP; i++ {
		squared_row := make([][]float64, CONV_SIZE)
		for j := 0; j < CONV_SIZE; j++ {
			squared_col := make([]float64, CONV_SIZE)
			for k := 0; k < CONV_SIZE; k++ {
				ct := conv_layer[i][j][k]
				sq_ct := ct * ct
				squared_col[k] = sq_ct
			}
			squared_row[j] = squared_col
		}
		squared[i] = squared_row
	}
	return squared
}

// Perform the pooling layer
//
// Weighted sums layer from convolution layer (after squaring)
func pool_layer(sq_layer_1 [][][]float64) []float64 {
	pool_kernels, pool_bias := prepare_pool_layer()

	outputs := make([]float64, POOL_LAYERS)
	for o := 0; o < POOL_LAYERS; o++ {
		sum := 0.
		index := 0
		for i := 0; i < CONV_MAP; i++ {
			for j := 0; j < CONV_SIZE; j++ {
				for k := 0; k < CONV_SIZE; k++ {
					sum += sq_layer_1[i][j][k] * pool_kernels[o][index]
					index++
				}
			}
		}
		outputs[o] = sum
	}

	for i := 0; i < POOL_LAYERS; i++ {
		outputs[i] += pool_bias[i]
	}

	return outputs
}

// Square all values
func square_layer_2(pool []float64) []float64 {
	squared := make([]float64, POOL_LAYERS)
	for i := 0; i < POOL_LAYERS; i++ {
		ct := pool[i]
		sq_ct := ct * ct
		squared[i] = sq_ct
	}
	return squared
}

// Perform the fc layer
//
// Weighted sums layer from pooling layer (after squaring)
func fc_layer(sq_layer_2 []float64) []float64 {
	fc_kernels, fc_bias := prepare_fc_layer()

	outputs := make([]float64, FC_LAYERS)
	for o := 0; o < FC_LAYERS; o++ {
		sum := 0.
		for i := 0; i < POOL_LAYERS; i++ {
			sum += sq_layer_2[i] * fc_kernels[o][i]
		}
		outputs[o] = sum
	}

	for i := 0; i < FC_LAYERS; i++ {
		outputs[i] += fc_bias[i]
	}

	return outputs
}

// Run cryptonets for a single image
func cryptonets(image [][]float64) int {
	// 1. Convolution layer
	conv_layer := convolution_layer(image)

	// 2. Square activation layer
	sq_layer_1 := square_layer_1(conv_layer)

	// 3. Pool layer
	pool := pool_layer(sq_layer_1)

	// 4. Square activation layer
	sq_layer_2 := square_layer_2(pool)

	// 5. Fully connected layer
	output := fc_layer(sq_layer_2)

	res := 0.
	index := 0
	for i := 1; i < FC_LAYERS; i++ {
		if output[i] > res {
			res = output[i]
			index = i
		}
	}

	return index
}

// Print padded image
func print_padded_image(image [][]float64) {
	for i := 0; i < PADDED_IMAGE_WIDTH; i++ {
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			fmt.Print(image[i][j], " ")
		}
		fmt.Println()
	}
}

// Print image after convolutions
func print_image_after_convolution(image [][][]float64) {
	for i := 0; i < CONV_MAP; i++ {
		for j := 0; j < CONV_SIZE; j++ {
			for k := 0; k < CONV_SIZE; k++ {
				fmt.Print(image[i][j][k], " ")
			}
		}
	}
	fmt.Println()
}

// Print image after pool layer
func print_image_after_pool_layer(image []float64) {
	for i := 0; i < POOL_LAYERS; i++ {
		fmt.Print(image[i], " ")
	}
	fmt.Println()
}

// Print image after fc layer
func print_image_after_fc_layer(image []float64) {
	for i := 0; i < FC_LAYERS; i++ {
		fmt.Print(image[i], " ")
	}
	fmt.Println()
}

// Print image as binary values
func print_binary_image(image [][]float64) {
	for i := 0; i < PADDED_IMAGE_WIDTH; i++ {
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			if image[i][j] > 0 {
				fmt.Print("1 ")
			} else {
				fmt.Print("0 ")
			}
		}
		fmt.Println()
	}
	fmt.Println()
}

// Print image as normalized values
func print_image(image [][]float64) {
	for i := 0; i < PADDED_IMAGE_WIDTH; i++ {
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			fmt.Print(float64(image[i][j]), " ")
		}
		fmt.Println()
	}
}

func main() {
	images, answers := get_images()

	errs := 0
	for i, image := range images {
		predicted := cryptonets(image)
		answer := answers[i]

		if predicted != answer {
			errs++
		}

		if i > 0 && i%100 == 0 {
			fmt.Printf("errs %d/%d \t", errs, i)
			fmt.Println("accuracy", 100.-(100.*float64(errs)/float64(i)))
		}
	}

	fmt.Println(" =========== Final results =========== ")
	fmt.Printf("errs %d/%d \t", errs, SLOTS)
	fmt.Println("accuracy", 100.-(100.*float64(errs)/float64(SLOTS)))
}
