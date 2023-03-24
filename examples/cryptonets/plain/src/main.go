package main

import (
	"fmt"
	"math"
	"plain/src/mnist"
	"plain/src/model"
)

// Paper source: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/04/CryptonetsTechReport.pdf

const (
	IMAGE_WIDTH      = 28
	IMAGE_CROP       = 13
	CONVOLUTION_SIZE = IMAGE_CROP * IMAGE_CROP // 169
)

func pad_image(input [][]float64) [][]float64 {
	// Pad the upper and left edge with 0s
	padded_width := IMAGE_WIDTH + 1
	output := make([][]float64, padded_width)
	for i := 0; i < IMAGE_WIDTH; i++ {
		row := make([]float64, padded_width)
		for j := 0; j < IMAGE_WIDTH; j++ {
			row[j] = input[i][j]
		}
		row[IMAGE_WIDTH] = float64(0)
		output[i] = row
	}
	output[IMAGE_WIDTH] = make([]float64, padded_width)
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
	var answers []int = make([]int, 8192)

	// pack a matrix of 8192 x 784
	// pad image size from 784 to 841
	for r := 0; r < 8192; r++ {
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
	// scale: 16
	for i := 0; i < 8192; i++ {
		for j := 0; j < 29; j++ {
			for k := 0; k < 29; k++ {
				images[i][j][k] /= 16
			}
		}
	}

	return images, answers
}

func round(x, unit float64) float64 {
	return math.Round(x/unit) * unit
}

func prepare_conv() ([][]float64, []float64) {
	// rescale kernels
	// scale: 32
	rescaled_kernels := make([][]float64, 5)
	for k, kernel := range model.ConvKernels {
		rescaled_kernel := make([]float64, 25)
		for i := 0; i < 5; i++ {
			for j := 0; j < 5; j++ {
				val := kernel[i*5+j]
				rescaled_kernel[i*5+j] = float64(val)
			}
		}
		rescaled_kernels[k] = rescaled_kernel
	}

	// encode conv bias
	bias := make([]float64, 5)
	for i := 0; i < 5; i++ {
		bias[i] = model.ConvBias[i]
	}

	return rescaled_kernels, bias
}

func prepare_pool_layer() ([][]float64, []float64) {
	// len(weights) = 84500
	pool_layers := make([][]float64, 100)
	for i := 0; i < 100; i++ {
		pool_layer := make([]float64, 845)
		for j := 0; j < 845; j++ {
			// rescale pool layer
			val := model.Weights_1[i+j*100]
			// transpose weights
			pool_layer[j] = float64(val)
		}
		pool_layers[i] = pool_layer
	}

	pool_bias := make([]float64, 100)
	for i := 0; i < 100; i++ {
		val := model.PoolBias[i]
		pool_bias[i] = float64(val)
	}
	return pool_layers, pool_bias
}

func prepare_fc_layer() ([][]float64, []float64) {
	fc_layers := make([][]float64, 10)
	for i := 0; i < 10; i++ {
		fc_layer := make([]float64, 100)
		for j := 0; j < 100; j++ {
			// rescale fc layer
			val := model.Weights_3[i*100+j]
			// transpose weights
			fc_layer[j] = float64(val)
		}
		fc_layers[i] = fc_layer
	}

	fc_bias := make([]float64, 10)
	for i := 0; i < 10; i++ {
		val := model.FcBias[i]
		fc_bias[i] = float64(val)
	}

	return fc_layers, fc_bias
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

	outputs := make([][][]float64, 5)
	for o := 0; o < 5; o++ {
		row_output := make([][]float64, 13)
		windows := make([][]float64, 169)
		stride := 2

		for stride_y := 0; stride_y < 13; stride_y++ {
			col_output := make([]float64, 13)
			for stride_x := 0; stride_x < 13; stride_x++ {
				// create window
				window := make([]float64, 25)
				offset_x := stride_x * stride
				offset_y := stride_y * stride
				fmt.Println("corner:", offset_y, " ", offset_x)
				for i := 0; i < 5; i++ {
					for j := 0; j < 5; j++ {
						// padded image width is 29
						window[i*5+j] = image[j+offset_y][i+offset_x]
					}
				}
				fmt.Println("window: ", window)
				// fmt.Println()
				windows[stride_y*13+stride_x] = window

				// kernel is already in column major form
				kernel := conv_kernels[o]

				product := make([]float64, 25)
				for i := 0; i < 25; i++ {
					product[i] = window[i] * kernel[i]
				}

				// sum all products together
				sum := product[0]
				for i := 1; i < 25; i++ {
					sum += product[i]
				}

				// rescale sum
				sum /= 16.

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
	squared := make([][][]float64, 5)
	for i := 0; i < 5; i++ {
		squared_row := make([][]float64, 13)
		for j := 0; j < 13; j++ {
			squared_col := make([]float64, 13)
			for k := 0; k < 13; k++ {
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

	pool_layers := 100
	outputs := make([]float64, pool_layers)
	for o := 0; o < pool_layers; o++ {
		sum := float64(0)
		index := 0
		for i := 0; i < 5; i++ {
			for j := 0; j < 13; j++ {
				for k := 0; k < 13; k++ {
					sum += float64(sq_layer_1[i][j][k] * pool_kernels[o][index])
					index++
				}
			}
		}
		outputs[o] = sum
	}

	for i := 0; i < 100; i++ {
		outputs[i] += pool_bias[i]
	}

	return outputs
}

func square_layer_2(pool []float64) []float64 {
	squared := make([]float64, 100)
	for i := 0; i < 100; i++ {
		ct := pool[i]
		sq_ct := ct * ct
		squared[i] = sq_ct
	}
	return squared
}

func fc_layer(sq_layer_2 []float64) []float64 {
	fc_kernels, fc_bias := prepare_fc_layer()

	fc_layers := 10
	outputs := make([]float64, fc_layers)
	for o := 0; o < fc_layers; o++ {
		sum := float64(0)
		for i := 0; i < 100; i++ {
			sum += float64(sq_layer_2[i] * fc_kernels[o][i])
		}
		outputs[o] = sum
	}

	for i := 0; i < 10; i++ {
		outputs[i] += fc_bias[i]
	}

	return outputs
}

func cryptonets(image [][]float64) []float64 {
	fmt.Println("Cryptonets!")
	// Inference layers (see Table 1 of `Paper source`)

	// 1. Convolution layer
	// conv_layer := convolution_layer(image)

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

func main() {
	// Get batched images
	images, answers := get_images()

	image := images[0]
	answer := answers[0]
	for i := 0; i < 29; i++ {
		for j := 0; j < 29; j++ {
			if image[i][j] > 0 {
				fmt.Print("1 ")
			} else {
				fmt.Print("0 ")
			}
		}
		fmt.Println()
	}
	fmt.Println()
	for i := 0; i < 29; i++ {
		for j := 0; j < 29; j++ {
			fmt.Print(float64(image[i][j]), " ")
		}
		fmt.Println()
	}
	fmt.Println()
	fmt.Println("answer:", answer)
	fmt.Println()

	// perform cryptonets
	predicted_images := cryptonets(image)

	// decoded_images
	fmt.Println(" ========================================= ")
	fmt.Println(" ============ Final Output =============== ")
	fmt.Println(" ========================================= ")

	// // print encrypted layer
	// for i := 0; i < 29; i++ {
	// 	for j := 0; j < 29; j++ {
	// 		fmt.Print(float64(image[i*29+j])/16, " ")
	// 	}
	// 	fmt.Println()
	// }

	// // print conv layer
	// for i := 0; i < 5; i++ {
	// 	for j := 0; j < 13; j++ {
	// 		for k := 0; k < 13; k++ {
	// 			data := float64(predicted_images[i][j][k])
	// 			fmt.Print(data, " ")
	// 		}
	// 	}
	// }

	// print fc layer
	for i := 0; i < 10; i++ {
		data := float64(predicted_images[i])
		fmt.Print(data, " ")
	}
	fmt.Println()
}
