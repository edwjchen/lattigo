package main

import (
	"cryptonets/src/mnist"
	"cryptonets/src/model"
	"errors"
	"fmt"
	"math"

	"github.com/tuneinsight/lattigo/v4/bfv"
	"github.com/tuneinsight/lattigo/v4/rlwe"
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

// func convolution(input []float64, kernel []float64) []float64 {
// 	output := make([]float64, CONVOLUTION_SIZE)
// 	for y := 0; y < IMAGE_CROP; y++ {
// 		for x := 0; x < IMAGE_CROP; x++ {
// 			val := 0.
// 			for wy := 0; wy < WINDOW_WIDTH; wy++ {
// 				for wx := 0; wx < WINDOW_WIDTH; wx++ {
// 					kernel_pos := wx + wy*WINDOW_WIDTH
// 					val += kernel[kernel_pos] * input[(x*STRIDE+wx)+(y*STRIDE+wy)*IMAGE_CROP]
// 				}
// 			}
// 			output[x+y*IMAGE_CROP] = val
// 		}
// 	}
// 	return output
// }

// func convolution_layer(input []float64, weights []float64) []float64 {
// 	output := make([]float64, OUTPUT_CHANNELS*CONVOLUTION_SIZE)
// 	kernel_size := WINDOW_WIDTH * WINDOW_WIDTH
// 	// kernel := make([][]float64, kernel_size)

// 	for o := 0; o < OUTPUT_CHANNELS; o++ {
// 		for i := 0; i < kernel_size; i++ {

// 		}
// 	}

// 	// for o := 0; o < OUTPUT_CHANNELS; o++ {
// 	// 	// make kernel from weights
// 	// 	for i := 0; i < kernel_size; i++ {
// 	// 		kernel[i] = weights[o*kernel_size+i]
// 	// 	}
// 	// 	conv := convolution(input, kernel)
// 	// 	for i := 0; i < CONVOLUTION_SIZE; i++ {
// 	// 		output[o*CONVOLUTION_SIZE+i] = conv[i]
// 	// 	}
// 	// }
// 	return output
// }

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

// func cryptonets(input []float64) ([]float64, error) {
// 	// Inference layers (see Table 1 of `Paper source`)

// 	// 1. Convolution layer
// 	// Weighted sums layer with windows of size 5×5, stride size of 2. From
// 	// each window, 5 different maps are computed and a padding is added to
// 	// the upper side and left side of each image.

// 	// This is a column major traversal through model.Weights_0.
// 	conv := convolution_layer(input, model.Weights_0)

// 	// 2. Square layer
// 	// Squares each of the 835 outputs of the convolution layer
// 	sq_1 := square(conv)

// 	// Transpose Weights_1
// 	weights_1 := transpose(model.Weights_1, OUTPUT_CHANNELS*CONVOLUTION_SIZE, FULLY_CONNECTED_WIDTH)

// 	// 3. Pool layer
// 	// Weighted sum layer that generates 100 outputs from the 835 outputs of
// 	// 1st the square layer
// 	pool, err := pool_layer(sq_1, weights_1)
// 	if err != nil {
// 		return pool, err
// 	}

// 	// 4. Square layer
// 	// Squares each of the 100 outputs of the pool layer
// 	sq_2 := square(pool)

// 	// 5. Output layer
// 	// Weighted sum that generates 10 outputs (corresponding to the 10 digits)
// 	// from the 100 outputs of the 2nd square layer
// 	output, err_2 := output_layer(sq_2, model.Weights_3)
// 	if err_2 != nil {
// 		return output, err_2
// 	}

// 	return output, nil
// }

// func convolution_layer(evaluator bfv.Evaluator, encrypted_images []*rlwe.Ciphertext, conv_kernels []*rlwe.Plaintext) {
// 	// Weighted sums layer with windows of size 5×5, stride size of 2. From
// 	// each window, 5 different maps are computed and a padding is added to
// 	// the upper side and left side of each image.

// 	// Create windows
// 	// encrypted images represents 8192 x 841 (padded)
// 	// where each row is an image and each column is a point

// 	outputs := make([][][]*rlwe.Ciphertext, 5)

// 	for o := 0; o < OUTPUT_CHANNELS; o++ {
// 		row_output := make([][]*rlwe.Ciphertext, 13)
// 		windows := make([][]*rlwe.Ciphertext, 169)

// 		stride := 2
// 		for stride_y := 0; stride_y < 13; stride_y++ {
// 			col_output := make([]*rlwe.Ciphertext, 13)

// 			for stride_x := 0; stride_x < 13; stride_x++ {
// 				window := make([]*rlwe.Ciphertext, 25)
// 				offset_y := stride_y * stride
// 				offset_x := stride_x * stride
// 				for i := 0; i < 5; i++ {
// 					for j := 0; j < 5; j++ {
// 						// column major order
// 						window[i*5+j] = encrypted_images[(i+offset_x)+(j*5+offset_y)]
// 					}
// 				}
// 				windows[stride_y*13+stride_x] = window

// 				// dot product
// 				kernel := conv_kernels[o]
// 				ct := bfv.NewCiphertext(params, window[0].Degree(), window[0].Level())
// 				// vector vector multiplication
// 				// evaluator.Mul()
// 				// col_output[stride_x] =
// 			}
// 		}

// 	}

// }

func params() bfv.Parameters {
	// Creating encryption parameters from a default params with N=8192
	paramsDef := bfv.PN13QP218
	paramsDef.T = 65537
	params, err := bfv.NewParametersFromLiteral(paramsDef)
	if err != nil {
		panic(err)
	}
	return params
}

func pad_image(input [][]uint64) [][]uint64 {
	// Pad the upper and left edge with 0s
	padded_width := IMAGE_WIDTH + 1
	output := make([][]uint64, padded_width)
	output[0] = make([]uint64, padded_width)
	for i := 0; i < IMAGE_WIDTH; i++ {
		row := make([]uint64, padded_width)
		row[0] = uint64(0)
		for j := 0; j < IMAGE_WIDTH; j++ {
			row[j+1] = input[i][j]
		}
		output[i+1] = row
	}
	return output
}

func get_images() [][]uint64 {
	// Load MNIST datasets
	dataSet, err := mnist.ReadTestSet("./mnist")
	if err != nil {
		fmt.Println(err)
		panic("MNIST Dataset error")
	}

	var images [][]uint64 = make([][]uint64, 0)

	// pack a matrix of 8192 x 784
	// pad image size from 784 to 841
	for r := 0; r < 8192; r++ {
		image := dataSet.Data[r]
		image_matrix := make([][]uint64, IMAGE_WIDTH)
		// flatten image into vector
		for i := 0; i < IMAGE_WIDTH; i++ {
			image_row := make([]uint64, IMAGE_WIDTH)
			for j := 0; j < IMAGE_WIDTH; j++ {
				image_row[j] = uint64(image.Image[i][j])
			}
			image_matrix[i] = image_row
		}
		padded_image := pad_image(image_matrix)

		// flatten padded image
		padded_width := IMAGE_WIDTH + 1
		flatten_image := make([]uint64, padded_width*padded_width)
		for i := 0; i < padded_width; i++ {
			for j := 0; j < padded_width; j++ {
				flatten_image[i*padded_width+j] = padded_image[i][j]
			}
		}
		images = append(images, flatten_image)
	}

	// normalize images
	// scale: 16
	for i := 0; i < 8192; i++ {
		for j := 0; j < 841; j++ {
			images[i][j] /= 16
		}
	}

	return images
}

func round(x, unit float64) float64 {
	return math.Round(x/unit) * unit
}

type CryptoNets struct {
	params    bfv.Parameters
	kgen      rlwe.KeyGenerator
	sk        *rlwe.SecretKey
	encoder   bfv.Encoder
	encryptor rlwe.Encryptor
	decryptor rlwe.Decryptor
	evaluator bfv.Evaluator
}

func (cn CryptoNets) encode(data []uint64) *rlwe.Plaintext {
	pt := bfv.NewPlaintext(cn.params, cn.params.MaxLevel())
	cn.encoder.Encode(data, pt)
	return pt
}

func (cn CryptoNets) encode_images(images [][]uint64) []*rlwe.Plaintext {
	// encode inputs into polynomials
	fmt.Println("Encoding!")
	encoded_images := make([]*rlwe.Plaintext, 841)
	for i := 0; i < 841; i++ {
		col := make([]uint64, 8192)
		for j := 0; j < 8192; j++ {
			col[j] = images[j][i]
		}
		pt := bfv.NewPlaintext(cn.params, cn.params.MaxLevel())
		cn.encoder.Encode(col, pt)
		encoded_images[i] = pt
	}
	return encoded_images
}

func (cn CryptoNets) encrypt_images(encoded_images []*rlwe.Plaintext) []*rlwe.Ciphertext {
	// encrypt
	fmt.Println("Encrypting!")
	encrypted_images := make([]*rlwe.Ciphertext, 841)
	for i := 0; i < 841; i++ {
		encrypted_images[i] = cn.encryptor.EncryptNew(encoded_images[i])
	}
	return encrypted_images
}

// func (cn CryptoNets) decrypt_images(encrypted_images []*rlwe.Ciphertext) []*rlwe.Plaintext {
// 	// decrypt
// 	fmt.Println("Decrypting!")
// 	decrypted_images := make([]*rlwe.Plaintext, 841)
// 	index := 0
// 	for i := 0; i < 841; i++ {
// 		pt := bfv.NewPlaintext(cn.params, encrypted_images[i].Level())
// 		cn.decryptor.Decrypt(encrypted_images[i], pt)
// 		decrypted_images[index] = pt
// 		index++
// 	}
// 	return decrypted_images
// }

func (cn CryptoNets) decrypt(encrypted *rlwe.Ciphertext) *rlwe.Plaintext {
	// decrypt
	fmt.Println("Decrypting!")
	pt := bfv.NewPlaintext(cn.params, encrypted.Level())
	cn.decryptor.Decrypt(encrypted, pt)
	return pt
}

func (cn CryptoNets) decrypt_images(encrypted_images [][][]*rlwe.Ciphertext) []*rlwe.Plaintext {
	// decrypt
	fmt.Println("Decrypting!")
	decrypted_images := make([]*rlwe.Plaintext, 845)
	index := 0
	for i := 0; i < 5; i++ {
		for j := 0; j < 13; j++ {
			for k := 0; k < 13; k++ {
				pt := bfv.NewPlaintext(cn.params, encrypted_images[i][j][k].Level())
				cn.decryptor.Decrypt(encrypted_images[i][j][k], pt)
				decrypted_images[index] = pt
				index++
			}
		}
	}
	return decrypted_images
}

func (cn CryptoNets) decode(decrypted *rlwe.Plaintext) []uint64 {
	// decode
	fmt.Println("Decoding!")
	decoded := make([]uint64, 8192)
	cn.encoder.Decode(decrypted, decoded)
	return decoded
}

func (cn CryptoNets) decode_images(decrypted_images []*rlwe.Plaintext) [][]uint64 {
	// decode
	fmt.Println("Decoding!")
	decoded_images := make([][]uint64, 845)
	for i := 0; i < 845; i++ {
		decoded_image := make([]uint64, 8192)
		cn.encoder.Decode(decrypted_images[i], decoded_image)
		decoded_images[i] = decoded_image
	}
	return decoded_images
}

func (cn CryptoNets) prepare_conv() ([][]*rlwe.Plaintext, []*rlwe.Plaintext) {
	// rescale kernels
	// scale: 32
	scale := 32.
	rescaled_kernels := make([][]uint64, 5)
	for k, kernel := range model.ConvKernels {
		rescaled_kernel := make([]uint64, 25)
		for i := 0; i < 5; i++ {
			for j := 0; j < 5; j++ {
				val := kernel[i*5+j] * scale
				if val < 0 {
					// TODO: rescaling kernels is slightly wrong since there are two plaintext
					// moduli, but we're only wrapping around thet first one
					val += float64(cn.params.T())
				}
				rescaled_kernel[i*5+j] = uint64(val)
			}
		}
		rescaled_kernels[k] = rescaled_kernel
	}

	// encode kernels
	// each weight is encoded as its own plaintext
	encoded_kernels := make([][]*rlwe.Plaintext, 5)
	for i := 0; i < 5; i++ {
		encoded_kernel := make([]*rlwe.Plaintext, 25)
		for j := 0; j < 25; j++ {
			val := rescaled_kernels[i][j]
			kernel := make([]uint64, 8192)
			for k := 0; k < 8192; k++ {
				kernel[k] = val
			}
			encoded_kernel[j] = cn.encode(kernel)
		}
		encoded_kernels[i] = encoded_kernel
	}

	// encode conv bias
	encoded_bias := make([]*rlwe.Plaintext, 5)
	for i := 0; i < 5; i++ {
		val := model.ConvBias[i]
		// rescale val
		val *= scale
		if val < 0 {
			val += float64(cn.params.T())
		}
		bias := make([]uint64, 8192)
		for j := 0; j < 8192; j++ {
			bias[j] = uint64(val)
		}
		encoded_bias[i] = cn.encode(bias)
	}

	return encoded_kernels, encoded_bias
}

func (cn CryptoNets) convolution_layer(encrypted_images []*rlwe.Ciphertext) [][][]*rlwe.Ciphertext {
	// Weighted sums layer with windows of size 5×5, stride size of 2. From
	// each window, 5 different maps are computed and a padding is added to
	// the upper side and left side of each image.

	// Create windows
	// encrypted_images represents 8192 x 841 (padded)
	// where each row is an image and each column is a point

	// Prepare model weights / bias
	conv_kernels, conv_bias := cn.prepare_conv()

	outputs := make([][][]*rlwe.Ciphertext, 5)
	for o := 0; o < OUTPUT_CHANNELS; o++ {
		row_output := make([][]*rlwe.Ciphertext, 13)
		windows := make([][]*rlwe.Ciphertext, 169)
		stride := 2

		for stride_y := 0; stride_y < 13; stride_y++ {
			col_output := make([]*rlwe.Ciphertext, 13)
			for stride_x := 0; stride_x < 13; stride_x++ {
				// create window
				window := make([]*rlwe.Ciphertext, 25)
				offset_y := stride_y * stride
				offset_x := stride_x * stride
				for i := 0; i < 5; i++ {
					for j := 0; j < 5; j++ {
						// padded image width is 29
						// column major order
						window[i*5+j] = encrypted_images[((i+offset_y)*29)+(j+offset_x)]
					}
				}
				windows[stride_y*13+stride_x] = window

				// dot product
				product := make([]*rlwe.Ciphertext, 25)
				kernel := conv_kernels[o]

				// decoded_kernel := cn.decode(kernel)
				// fmt.Println("decoded kernel")
				// for i := 0; i < 8192; i++ {
				// 	fmt.Print(decoded_kernel[i], " ")
				// }
				// fmt.Println()

				for i := 0; i < 25; i++ {
					ct := bfv.NewCiphertext(cn.params, window[0].Degree(), window[0].Level())

					// decode_window := cn.decode(cn.decrypt(window[i]))
					// sum := uint64(0)
					// for i := 0; i < 8192; i++ {
					// 	sum += decode_window[i]
					// }

					cn.evaluator.Mul(window[i], kernel[i], ct)

					// if sum > 0 {
					// 	res := cn.decode(cn.decrypt(ct))
					// 	fmt.Println("product result")
					// 	for i := 0; i < 8192; i++ {
					// 		fmt.Print(res[i], " ")
					// 	}
					// 	fmt.Println()
					// } else {
					// 	fmt.Println("window is 0")
					// }

					product[i] = ct
				}

				// sum all products together
				sum := product[0]
				for i := 1; i < 25; i++ {
					cn.evaluator.Add(sum, product[i], sum)
				}

				// res := cn.decode(cn.decrypt(sum))
				// fmt.Println("sum!")
				// for i := 0; i < 8192; i++ {
				// 	fmt.Print(res[i], " ")
				// }
				// fmt.Println()

				col_output[stride_x] = sum
			}
			row_output[stride_y] = col_output
		}
		outputs[o] = row_output
	}

	// add bias
	index := 0
	for i := 0; i < 5; i++ {
		for j := 0; j < 13; j++ {
			for k := 0; k < 13; k++ {
				bias := conv_bias[i]
				index++
				ct := outputs[i][j][k]
				cn.evaluator.Add(ct, bias, ct)
				outputs[i][j][k] = ct
			}
		}
	}

	return outputs
}

func (cn CryptoNets) cryptonets(encrypted_images []*rlwe.Ciphertext) [][][]*rlwe.Ciphertext {
	fmt.Println("Cryptonets!")
	// Inference layers (see Table 1 of `Paper source`)

	// 1. Convolution layer
	outputs := cn.convolution_layer(encrypted_images)
	return outputs
}

func main() {
	// Get batched images
	images := get_images()

	// Initialize encryption parameters
	params := params()
	kgen := bfv.NewKeyGenerator(params)
	sk, _ := kgen.GenKeyPair()
	encoder := bfv.NewEncoder(params)
	encryptor_sk := bfv.NewEncryptor(params, sk)
	decryptor := bfv.NewDecryptor(params, sk)
	evaluator := bfv.NewEvaluator(params, rlwe.EvaluationKey{})

	cn := CryptoNets{
		params, kgen, sk, encoder, encryptor_sk, decryptor, evaluator,
	}

	// encode
	encoded_images := cn.encode_images(images)

	// encrypt
	encrypted_images := cn.encrypt_images(encoded_images)

	// perform cryptonets
	predicted_images := cn.cryptonets(encrypted_images)

	// decrypt
	decrypted_images := cn.decrypt_images(predicted_images)

	// decode
	decoded_images := cn.decode_images(decrypted_images)

	// decoded_images
	fmt.Println(" ========================================= ")
	fmt.Println(" ============ Final Output =============== ")
	fmt.Println(" ========================================= ")
	for i := 0; i < 8192; i++ {
		for j := 0; j < 845; j++ {
			fmt.Print(decoded_images[j][i], " ")
		}
		fmt.Println()
	}
}
