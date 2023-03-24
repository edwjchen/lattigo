package main

import (
	"fmt"
	"math"
	"plain/src/mnist"
	"plain/src/model"
)

// Paper source: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/04/CryptonetsTechReport.pdf

const (
	SLOTS              = 8192
	IMAGE_WIDTH        = 28
	PADDED_IMAGE_WIDTH = IMAGE_WIDTH + 1
	IMAGE_NORM         = 256
	STRIDE             = 2
	CONV_MAP           = 5
	CONV_SIZE          = 13
	CONV_WINDOW_LEN    = 5 // 5 x 5
	CONVOLUTION_SIZE   = 845
	POOL_LAYERS        = 100
	FC_LAYERS          = 10
)

func pad_image(input [][]float64) [][]float64 {
	// Pad the upper and left edge with 0s
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

func get_images() ([][][]float64, []int) {
	// Load MNIST datasets
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
		// flatten image into vector
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

func round(x, unit float64) float64 {
	return math.Round(x/unit) * unit
}

func prepare_conv() ([][]float64, []float64) {
	return model.ConvKernels, model.ConvBias
}

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

func convolution_layer(image [][]float64) [][][]float64 {
	// Weighted sums layer with windows of size 5×5, stride size of 2. From
	// each window, 5 different maps are computed and a padding is added to
	// the upper side and left side of each image.

	// Create windows
	// encrypted_images represents 8192 x 841 (padded)
	// where each row is an image and each column is a point

	// Prepare model weights / bias
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

				// kernel is already in column major form
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

func square_layer_2(pool []float64) []float64 {
	squared := make([]float64, POOL_LAYERS)
	for i := 0; i < POOL_LAYERS; i++ {
		ct := pool[i]
		sq_ct := ct * ct
		squared[i] = sq_ct
	}
	return squared
}

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

func cryptonets(image [][]float64) []float64 {
	fmt.Println("Cryptonets!")

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

	return output
}

func decode_encrypted_layer(image [][]float64) {
	for i := 0; i < PADDED_IMAGE_WIDTH; i++ {
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			fmt.Print(image[i][j], " ")
		}
		fmt.Println()
	}
}

func decode_conv_layer(image [][][]float64) {
	for i := 0; i < 5; i++ {
		for j := 0; j < 13; j++ {
			for k := 0; k < 13; k++ {
				fmt.Print(image[i][j][k], " ")
			}
		}
	}
}

func decode_fc_layer(image []float64) {
	for i := 0; i < 10; i++ {
		fmt.Print(image[i], " ")
	}
}

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

func print_image(image [][]float64) {
	for i := 0; i < PADDED_IMAGE_WIDTH; i++ {
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			fmt.Print(float64(image[i][j]), " ")
		}
		fmt.Println()
	}
}

func main() {
	// Get batched images
	images, answers := get_images()

	image := images[0]
	answer := answers[0]
	print_binary_image(image)
	print_image(image)
	fmt.Println("answer:", answer)
	fmt.Println()

	// perform cryptonets
	predicted_images := cryptonets(image)

	// print fc layer
	for i := 0; i < 10; i++ {
		data := float64(predicted_images[i])
		fmt.Print(data, " ")
	}
	fmt.Println()
}
