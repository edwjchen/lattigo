package main

import (
	"encrypted/src/mnist"
	"encrypted/src/model"
	"fmt"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

// Paper source: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/04/CryptonetsTechReport.pdf

const (
	SLOTS              = 8192                                    // number of slots within a plaintext
	IMAGE_WIDTH        = 28                                      // width of image
	PADDED_IMAGE_WIDTH = IMAGE_WIDTH + 1                         // width of image after padding
	PADDED_IMAGE_SIZE  = PADDED_IMAGE_WIDTH * PADDED_IMAGE_WIDTH // padded image size
	IMAGE_NORM         = 256                                     // normalization scale for images
	STRIDE             = 2                                       // stride of convolution window
	CONV_MAP           = 5                                       // number of convolution maps
	CONV_SIZE          = 13                                      // convolution width (and length)
	CONV_WINDOW_LEN    = 5                                       // size of convolution window
	CONVOLUTION_SIZE   = 845                                     // number of convolution layers
	POOL_LAYERS        = 100                                     // number of pooling layers
	FC_LAYERS          = 10                                      // number of fc layers
)

// Pad the right and bottom edges of an image with 0s
func pad_image(input [][]float64) [][]float64 {
	output := make([][]float64, PADDED_IMAGE_WIDTH)
	for i := 0; i < IMAGE_WIDTH; i++ {
		row := make([]float64, PADDED_IMAGE_WIDTH)
		for j := 0; j < IMAGE_WIDTH; j++ {
			row[j] = input[i][j]
		}
		row[IMAGE_WIDTH] = 0.
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

	var images [][][]float64 = make([][][]float64, SLOTS)
	var answers []int = make([]int, SLOTS)

	// pack a matrix of 8192 x 784
	// pad image size from 784 to 841
	for i := 0; i < SLOTS; i++ {
		image := dataSet.Data[i]
		answers[i] = image.Digit
		image_matrix := make([][]float64, IMAGE_WIDTH)
		for j := 0; j < IMAGE_WIDTH; j++ {
			image_row := make([]float64, IMAGE_WIDTH)
			for k := 0; k < IMAGE_WIDTH; k++ {
				image_row[k] = float64(image.Image[j][k])
			}
			image_matrix[j] = image_row
		}
		padded_image := pad_image(image_matrix)
		images[i] = padded_image
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

func (cn CryptoNets) rescale_images(images [][][]float64) [][][]float64 {
	rescaled_images := make([][][]float64, SLOTS)
	var scale float64 = 1.
	for i := 0; i < SLOTS; i++ {
		rescaled_image := make([][]float64, PADDED_IMAGE_WIDTH)
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			rescaled_col := make([]float64, PADDED_IMAGE_WIDTH)
			for k := 0; k < PADDED_IMAGE_WIDTH; k++ {
				rescaled_col[k] = images[i][j][k] * scale
			}
			rescaled_image[j] = rescaled_col
		}
		rescaled_images[i] = rescaled_image
	}

	return rescaled_images
}

// Get encryption parameters from a default params with N=8192
func params() ckks.Parameters {
	paramsDef := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	if err != nil {
		panic(err)
	}
	return params
}

type CryptoNets struct {
	params    ckks.Parameters
	kgen      rlwe.KeyGenerator
	sk        *rlwe.SecretKey
	encoder   ckks.Encoder
	encryptor rlwe.Encryptor
	decryptor rlwe.Decryptor
	evaluator ckks.Evaluator
}

func (cn CryptoNets) encode(data []complex128) *rlwe.Plaintext {
	pt := cn.encoder.EncodeNew(data, cn.params.MaxLevel(), cn.params.DefaultScale(), cn.params.LogSlots())
	return pt
}

func (cn CryptoNets) encode_images(images [][][]float64) [][]*rlwe.Plaintext {
	encoded_images := make([][]*rlwe.Plaintext, PADDED_IMAGE_WIDTH)
	for i := 0; i < PADDED_IMAGE_WIDTH; i++ {
		encoded_col := make([]*rlwe.Plaintext, PADDED_IMAGE_WIDTH)
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			pixel := make([]complex128, SLOTS)
			for k := 0; k < SLOTS; k++ {
				pixel[k] = complex(images[k][i][j], 0)
			}
			encoded_pixel := cn.encode(pixel)
			encoded_col[j] = encoded_pixel
		}
		encoded_images[i] = encoded_col
	}
	return encoded_images
}

func (cn CryptoNets) encrypt(image *rlwe.Plaintext) *rlwe.Ciphertext {
	return cn.encryptor.EncryptNew(image)
}

func (cn CryptoNets) encrypt_images(encoded_images [][]*rlwe.Plaintext) [][]*rlwe.Ciphertext {
	encrypted_image := make([][]*rlwe.Ciphertext, PADDED_IMAGE_WIDTH)
	for i := 0; i < PADDED_IMAGE_WIDTH; i++ {
		encrypted_col := make([]*rlwe.Ciphertext, PADDED_IMAGE_WIDTH)
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			encrypted_col[j] = cn.encrypt(encoded_images[i][j])
		}
		encrypted_image[i] = encrypted_col
	}
	return encrypted_image
}

func (cn CryptoNets) decrypt(encrypted *rlwe.Ciphertext) *rlwe.Plaintext {
	return cn.decryptor.DecryptNew(encrypted)
}

func (cn CryptoNets) decrypt_images(encrypted_image [][]*rlwe.Ciphertext) [][]*rlwe.Plaintext {
	decrypted_image := make([][]*rlwe.Plaintext, PADDED_IMAGE_WIDTH)
	for i := 0; i < PADDED_IMAGE_WIDTH; i++ {
		decrypted_col := make([]*rlwe.Plaintext, PADDED_IMAGE_WIDTH)
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			decrypted_col[j] = cn.decrypt(encrypted_image[i][j])
		}
		decrypted_image[i] = decrypted_col
	}
	return decrypted_image
}

func (cn CryptoNets) decrypt_conv(encrypted_image [][][]*rlwe.Ciphertext) [][][]*rlwe.Plaintext {
	decrypted_image := make([][][]*rlwe.Plaintext, CONV_MAP)
	for i := 0; i < CONV_MAP; i++ {
		decrypted_row := make([][]*rlwe.Plaintext, CONV_SIZE)
		for j := 0; j < CONV_SIZE; j++ {
			decrypted_col := make([]*rlwe.Plaintext, CONV_SIZE)
			for k := 0; k < CONV_SIZE; k++ {
				decrypted_col[k] = cn.decrypt(encrypted_image[i][j][k])
			}
			decrypted_row[j] = decrypted_col
		}
		decrypted_image[i] = decrypted_row
	}
	return decrypted_image
}

func (cn CryptoNets) decrypt_pool(encrypted_image []*rlwe.Ciphertext) []*rlwe.Plaintext {
	decrypted_image := make([]*rlwe.Plaintext, POOL_LAYERS)
	for i := 0; i < POOL_LAYERS; i++ {
		decrypted_image[i] = cn.decrypt(encrypted_image[i])
	}
	return decrypted_image
}

func (cn CryptoNets) decrypt_fc(encrypted_image []*rlwe.Ciphertext) []*rlwe.Plaintext {
	decrypted_image := make([]*rlwe.Plaintext, FC_LAYERS)
	for i := 0; i < FC_LAYERS; i++ {
		decrypted_image[i] = cn.decrypt(encrypted_image[i])
	}
	return decrypted_image
}

func (cn CryptoNets) decode(decrypted *rlwe.Plaintext) []complex128 {
	// decode
	return cn.encoder.Decode(decrypted, cn.params.LogSlots())
}

func (cn CryptoNets) decode_images(decrypted_images [][]*rlwe.Plaintext) [][][]float64 {
	// decode
	decoded_images := make([][][]float64, PADDED_IMAGE_WIDTH)
	for i := 0; i < PADDED_IMAGE_WIDTH; i++ {
		decoded_col := make([][]float64, PADDED_IMAGE_WIDTH)
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			decoded_image := cn.decode(decrypted_images[i][j])
			decoded_pixel := make([]float64, SLOTS)
			for k := 0; k < len(decoded_image); k++ {
				decoded_pixel[k] = real(decoded_image[k])
			}
			decoded_col[j] = decoded_pixel
		}
		decoded_images[i] = decoded_col
	}
	return decoded_images
}

func (cn CryptoNets) decode_conv(decrypted_images [][][]*rlwe.Plaintext) [][][][]float64 {
	// decode
	decoded_images := make([][][][]float64, CONV_MAP)
	for i := 0; i < CONV_MAP; i++ {
		decoded_row := make([][][]float64, CONV_SIZE)
		for j := 0; j < CONV_SIZE; j++ {
			decoded_col := make([][]float64, CONV_SIZE)
			for k := 0; k < CONV_SIZE; k++ {
				decoded_real := make([]float64, SLOTS)
				decoded := cn.decode(decrypted_images[i][j][k])
				for l := 0; l < SLOTS; l++ {
					decoded_real[l] = real(decoded[l])
				}
				decoded_col[k] = decoded_real
			}
			decoded_row[j] = decoded_col
		}
		decoded_images[i] = decoded_row
	}
	return decoded_images
}

func (cn CryptoNets) decode_pool(encoded_image []*rlwe.Plaintext) [][]float64 {
	decoded_image := make([][]float64, POOL_LAYERS)
	for i := 0; i < POOL_LAYERS; i++ {
		decoded_real := make([]float64, SLOTS)
		decoded := cn.decode(encoded_image[i])
		for j := 0; j < SLOTS; j++ {
			decoded_real[j] = real(decoded[j])
		}
		decoded_image[i] = decoded_real
	}
	return decoded_image
}

func (cn CryptoNets) decode_fc(encoded_image []*rlwe.Plaintext) [][]float64 {
	decoded_image := make([][]float64, FC_LAYERS)
	for i := 0; i < FC_LAYERS; i++ {
		decoded_real := make([]float64, SLOTS)
		decoded := cn.decode(encoded_image[i])
		for j := 0; j < SLOTS; j++ {
			decoded_real[j] = real(decoded[j])
		}
		decoded_image[i] = decoded_real
	}
	return decoded_image
}

// func (cn CryptoNets) prepare_conv() ([][]*rlwe.Plaintext, []*rlwe.Plaintext) {
// 	// rescale kernels
// 	kernel_scale := 262144.
// 	rescaled_kernels := make([][]float64, 5)
// 	for k, kernel := range model.ConvKernels {
// 		rescaled_kernel := make([]float64, 25)
// 		for i := 0; i < 5; i++ {
// 			for j := 0; j < 5; j++ {
// 				// rescale kernel
// 				val := kernel[i*5+j] * kernel_scale
// 				rescaled_kernel[i*5+j] = float64(val)
// 			}
// 		}
// 		rescaled_kernels[k] = rescaled_kernel
// 	}

// 	// encode kernels
// 	// each weight is encoded as its own plaintext
// 	encoded_kernels := make([][]*rlwe.Plaintext, 5)
// 	for i := 0; i < 5; i++ {
// 		encoded_kernel := make([]*rlwe.Plaintext, 25)
// 		for j := 0; j < 25; j++ {
// 			val := rescaled_kernels[i][j]
// 			kernel := make([]float64, 8192)
// 			for k := 0; k < 8192; k++ {
// 				kernel[k] = val
// 			}
// 			encoded_kernel[j] = cn.encode(kernel)
// 		}
// 		encoded_kernels[i] = encoded_kernel
// 	}

// 	// encode conv bias
// 	bias_scale := 262144. * 262144.
// 	encoded_bias := make([]*rlwe.Plaintext, 5)
// 	for i := 0; i < 5; i++ {
// 		val := model.ConvBias[i]
// 		// rescale val
// 		val *= bias_scale
// 		bias := make([]float64, 8192)
// 		for j := 0; j < 8192; j++ {
// 			bias[j] = float64(val)
// 		}
// 		encoded_bias[i] = cn.encode(bias)
// 	}

// 	return encoded_kernels, encoded_bias
// }

func (cn CryptoNets) prepare_conv_scalar() ([][]float64, []float64) {
	// rescale kernels
	kernel_scale := 1.
	rescaled_kernels := make([][]float64, 5)
	for k, kernel := range model.ConvKernels {
		rescaled_kernel := make([]float64, 25)
		for i := 0; i < 5; i++ {
			for j := 0; j < 5; j++ {
				// rescale kernel
				val := kernel[i*5+j] * kernel_scale
				rescaled_kernel[i*5+j] = float64(val)
			}
		}
		rescaled_kernels[k] = rescaled_kernel
	}

	// encode conv bias
	bias_scale := 1.
	rescale_bias := make([]float64, 5)
	for i := 0; i < 5; i++ {
		val := model.ConvBias[i]
		// rescale val
		val *= bias_scale
		rescale_bias[i] = float64(val)
	}

	return rescaled_kernels, rescale_bias
}

func (cn CryptoNets) convolution_layer(encrypted_images [][]*rlwe.Ciphertext) [][][]*rlwe.Ciphertext {
	// Prepare model weights / bias
	conv_kernels, conv_bias := cn.prepare_conv_scalar()

	outputs := make([][][]*rlwe.Ciphertext, 5)
	for o := 0; o < 5; o++ {
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
						window[i*5+j] = encrypted_images[(j + offset_y)][(i + offset_x)]
					}
				}
				windows[stride_y*13+stride_x] = window

				// dot product
				product := make([]*rlwe.Ciphertext, 25)

				// kernel is already in column major form
				kernel := conv_kernels[o]

				for i := 0; i < 25; i++ {
					new_ct := cn.evaluator.MultByConstNew(window[i], kernel[i])
					if err := cn.evaluator.Rescale(new_ct, cn.params.DefaultScale(), new_ct); err != nil {
						panic(err)
					}
					product[i] = new_ct
				}

				// sum all products together
				sum := product[0]
				for i := 1; i < 25; i++ {
					cn.evaluator.Add(sum, product[i], sum)
				}

				// add bias
				bias := float64(conv_bias[o])
				sum = cn.evaluator.AddConstNew(sum, bias)

				col_output[stride_x] = sum

			}
			row_output[stride_y] = col_output
		}
		outputs[o] = row_output
	}

	return outputs
}

func (cn CryptoNets) square_layer_1(conv_layer [][][]*rlwe.Ciphertext) [][][]*rlwe.Ciphertext {
	squared := make([][][]*rlwe.Ciphertext, 5)
	for i := 0; i < 5; i++ {
		squared_row := make([][]*rlwe.Ciphertext, 13)
		for j := 0; j < 13; j++ {
			squared_col := make([]*rlwe.Ciphertext, 13)
			for k := 0; k < 13; k++ {
				ct := conv_layer[i][j][k]
				new_ct := cn.evaluator.MulNew(ct, ct)
				relin := cn.evaluator.RelinearizeNew(new_ct)
				if err := cn.evaluator.Rescale(relin, cn.params.DefaultScale(), relin); err != nil {
					panic(err)
				}
				// constant := cn.params.Q()[0]
				// // scale_up := cn.evaluator.ScaleUpNew(relin, rlwe.NewScale(constant))
				// fmt.Println("before relin.scale:", relin.Scale)
				// // scale up
				// relin = cn.evaluator.MultByConstNew(relin, constant)
				// fmt.Println("scaling relin.scale:", relin.Scale)
				// relin.Scale = relin.Scale.Mul(rlwe.NewScale(constant))
				// if err := cn.evaluator.Rescale(relin, relin.Scale, relin); err != nil {
				// 	fmt.Println("err:", err)
				// 	panic("Square layer 1: error with rescale")
				// }
				// fmt.Println("after relin.scale:", relin.Scale)
				// panic("todo")
				squared_col[k] = relin
			}
			squared_row[j] = squared_col
		}
		squared[i] = squared_row
	}
	return squared
}

// // func (cn CryptoNets) prepare_pool_layer() ([][]*rlwe.Plaintext, []*rlwe.Plaintext) {
// // 	// rescale kernels
// // 	// scale: 32
// // 	scale := 32.
// // 	pool_layers := make([][]float64, POOL_LAYERS)
// // 	for i := 0; i < POOL_LAYERS; i++ {
// // 		pool_layer := make([]float64, CONVOLUTION_SIZE)
// // 		for j := 0; j < CONVOLUTION_SIZE; j++ {
// // 			// transpose weights
// // 			val := model.Weights_1[i+j*POOL_LAYERS]
// // 			val *= scale
// // 			pool_layer[j] = val
// // 		}
// // 		pool_layers[i] = pool_layer
// // 	}

// // 	// encode
// // 	encoded_kernels := make([][]*rlwe.Plaintext, POOL_LAYERS)
// // 	for i := 0; i < POOL_LAYERS; i++ {
// // 		encoded_kernel := make([]*rlwe.Plaintext, CONVOLUTION_SIZE)
// // 		for j := 0; j < CONVOLUTION_SIZE; j++ {
// // 			val := pool_layers[i][j]
// // 			kernel := make([]float64, 8192)
// // 			for k := 0; k < 8192; k++ {
// // 				kernel[k] = float64(val)
// // 			}
// // 			encoded_kernel[j] = cn.encode(kernel)
// // 		}
// // 		encoded_kernels[i] = encoded_kernel
// // 	}

// // 	// encode conv bias
// // 	encoded_bias := make([]*rlwe.Plaintext, POOL_LAYERS)
// // 	for i := 0; i < POOL_LAYERS; i++ {
// // 		val := model.PoolBias[i]
// // 		// rescale val
// // 		val *= scale
// // 		bias := make([]float64, 8192)
// // 		for j := 0; j < 8192; j++ {
// // 			bias[j] = float64(val)
// // 		}
// // 		encoded_bias[i] = cn.encode(bias)
// // 	}

// // 	return encoded_kernels, encoded_bias
// // }

// func (cn CryptoNets) prepare_pool_kernel(o int, i int) *rlwe.Plaintext {
// 	// rescale kernels
// 	scale := 512.
// 	val := model.Weights_1[o+i*POOL_LAYERS]
// 	val *= scale
// 	// fmt.Println(o, i, val)

// 	// encode
// 	kernel := make([]float64, 8192)
// 	for k := 0; k < 8192; k++ {
// 		kernel[k] = float64(val)
// 	}

// 	return cn.encode(kernel)
// }

func (cn CryptoNets) prepare_pool_layer() ([]float64, []float64) {
	return model.Weights_1, model.PoolBias
}

// func (cn CryptoNets) prepare_pool_bias() []*rlwe.Plaintext {
// 	// rescale kernels
// 	scale := 512.

// 	// encode conv bias
// 	encoded_bias := make([]*rlwe.Plaintext, POOL_LAYERS)
// 	for i := 0; i < POOL_LAYERS; i++ {
// 		val := model.PoolBias[i]
// 		// rescale val
// 		val *= scale
// 		bias := make([]float64, 8192)
// 		for j := 0; j < 8192; j++ {
// 			bias[j] = float64(val)
// 		}
// 		encoded_bias[i] = cn.encode(bias)
// 	}

// 	return encoded_bias
// }

func (cn CryptoNets) pool_layer(sq_layer_1 [][][]*rlwe.Ciphertext) []*rlwe.Ciphertext {
	pool_kernels, pool_bias := cn.prepare_pool_layer()

	outputs := make([]*rlwe.Ciphertext, POOL_LAYERS)
	for o := 0; o < POOL_LAYERS; o++ {
		fmt.Println("layer:", o)
		index := 0

		products := make([]*rlwe.Ciphertext, CONVOLUTION_SIZE)
		for i := 0; i < CONV_MAP; i++ {
			for j := 0; j < CONV_SIZE; j++ {
				for k := 0; k < CONV_SIZE; k++ {
					ct := sq_layer_1[i][j][k]
					new_ct := cn.evaluator.MultByConstNew(ct, pool_kernels[o+index*POOL_LAYERS])
					if err := cn.evaluator.Rescale(new_ct, cn.params.DefaultScale(), new_ct); err != nil {
						panic(err)
					}
					// data := cn.decode(cn.decrypt(ct))[0]
					// kernel := cn.decode(cn.prepare_pool_kernel(o, index))[0]
					// mul := cn.decode(cn.decrypt(res))[0]
					// fmt.Println(index, data, kernel, mul)
					products[index] = new_ct
					index++
				}
			}
		}

		// for i := 0; i < CONVOLUTION_SIZE; i++ {
		// 	mul := cn.decode(cn.decrypt(products[i]))
		// 	fmt.Println(i, mul[0])
		// }

		sum := products[0]
		for i := 1; i < CONVOLUTION_SIZE; i++ {
			cn.evaluator.Add(sum, products[i], sum)
		}

		// add bias
		bias := pool_bias[o]
		sum = cn.evaluator.AddConstNew(sum, bias)

		outputs[o] = sum
	}
	return outputs
}

func (cn CryptoNets) square_layer_2(pool_layer []*rlwe.Ciphertext) []*rlwe.Ciphertext {
	squared := make([]*rlwe.Ciphertext, POOL_LAYERS)
	for i := 0; i < POOL_LAYERS; i++ {
		ct := pool_layer[i]
		new_ct := cn.evaluator.MulNew(ct, ct)
		relin := cn.evaluator.RelinearizeNew(new_ct)
		if err := cn.evaluator.Rescale(relin, cn.params.DefaultScale(), relin); err != nil {
			panic(err)
		}
		// constant := cn.params.Q()[1]
		// scale_up := cn.evaluator.ScaleUpNew(relin, rlwe.NewScale(constant))
		// if err := cn.evaluator.Rescale(scale_up, relin.Scale, scale_up); err != nil {
		// 	fmt.Println("err:", err)
		// 	panic("Square layer 1: error with rescale")
		// }
		squared[i] = relin
	}
	return squared
}

func (cn CryptoNets) prepare_fc_layer() ([]float64, []float64) {
	return model.Weights_3, model.FcBias
}

// func (cn CryptoNets) prepare_fc_layer() ([][]*rlwe.Plaintext, []*rlwe.Plaintext) {
// 	// rescale kernels
// 	scale := 128.
// 	fc_layers := make([][]float64, FC_LAYERS)
// 	for i := 0; i < FC_LAYERS; i++ {
// 		fc_layer := make([]float64, POOL_LAYERS)
// 		for j := 0; j < FC_LAYERS; j++ {
// 			// transpose weights
// 			val := model.Weights_3[i*POOL_LAYERS+j]
// 			val *= scale
// 			fc_layer[j] = val
// 		}
// 		fc_layers[i] = fc_layer
// 	}

// 	// encode
// 	encoded_kernels := make([][]*rlwe.Plaintext, FC_LAYERS)
// 	for i := 0; i < FC_LAYERS; i++ {
// 		encoded_kernel := make([]*rlwe.Plaintext, POOL_LAYERS)
// 		for j := 0; j < POOL_LAYERS; j++ {
// 			val := fc_layers[i][j]
// 			kernel := make([]float64, 8192)
// 			for k := 0; k < 8192; k++ {
// 				kernel[k] = float64(val)
// 			}
// 			encoded_kernel[j] = cn.encode(kernel)
// 		}
// 		encoded_kernels[i] = encoded_kernel
// 	}

// 	// encode conv bias
// 	encoded_bias := make([]*rlwe.Plaintext, FC_LAYERS)
// 	for i := 0; i < FC_LAYERS; i++ {
// 		val := model.PoolBias[i]
// 		// rescale val
// 		val *= scale
// 		bias := make([]float64, 8192)
// 		for j := 0; j < 8192; j++ {
// 			bias[j] = float64(val)
// 		}
// 		encoded_bias[i] = cn.encode(bias)
// 	}

// 	return encoded_kernels, encoded_bias

// }

func (cn CryptoNets) fc_layer(sq_layer_2 []*rlwe.Ciphertext) []*rlwe.Ciphertext {
	fc_kernels, fc_bias := cn.prepare_fc_layer()

	outputs := make([]*rlwe.Ciphertext, FC_LAYERS)
	for o := 0; o < FC_LAYERS; o++ {
		product := make([]*rlwe.Ciphertext, POOL_LAYERS)
		for i := 0; i < POOL_LAYERS; i++ {
			ct := sq_layer_2[i]
			new_ct := cn.evaluator.MultByConstNew(ct, fc_kernels[o*POOL_LAYERS+i])
			if err := cn.evaluator.Rescale(new_ct, cn.params.DefaultScale(), new_ct); err != nil {
				panic(err)
			}
			// fmt.Println("ct: ", real(cn.decode(cn.decrypt(ct))[0]))
			// fmt.Println("kernel: ", fc_kernels[o*POOL_LAYERS+i])
			// fmt.Println("new_ct: ", real(cn.decode(cn.decrypt(new_ct))[0]))
			// panic("todo")
			product[i] = new_ct
		}

		// sum all products together
		sum := product[0]
		for i := 1; i < POOL_LAYERS; i++ {
			cn.evaluator.Add(sum, product[i], sum)
		}

		// add bias
		bias := fc_bias[o]
		sum = cn.evaluator.AddConstNew(sum, bias)

		outputs[o] = sum
	}

	return outputs
}

func (cn CryptoNets) cryptonets(encrypted_images [][]*rlwe.Ciphertext) []*rlwe.Ciphertext {
	// Inference layers (see Table 1 of `Paper source`)

	// 1. Convolution layer
	fmt.Println("Convolution")
	conv_layer := cn.convolution_layer(encrypted_images)
	// return conv_layer

	// 2. Square activation layer
	fmt.Println("Square")
	sq_layer_1 := cn.square_layer_1(conv_layer)
	// return sq_layer_1

	// 3. Pooling layer
	fmt.Println("Pool")
	pool_layer := cn.pool_layer(sq_layer_1)
	// return pool_layer

	// 4. Square activation layer
	fmt.Println("Square 2")
	sq_layer_2 := cn.square_layer_2(pool_layer)
	// return sq_layer_2

	// 5. Fully connected layer
	fmt.Println("FC")
	output := cn.fc_layer(sq_layer_2)
	ct := output[0]
	fmt.Println("Done... Consumed levels:", cn.params.MaxLevel()-ct.Level())

	return output
}

func main() {
	// Initialize encryption parameters
	params := params()
	kgen := ckks.NewKeyGenerator(params)
	sk, _ := kgen.GenKeyPair()
	rlk := kgen.GenRelinearizationKey(sk, params.MaxLevel())
	encoder := ckks.NewEncoder(params)
	encryptor_sk := ckks.NewEncryptor(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	// evaluator := bfv.NewEvaluator(params, rlwe.EvaluationKey{})
	evaluator := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk})

	cn := CryptoNets{
		params, kgen, sk, encoder, encryptor_sk, decryptor, evaluator,
	}

	// Get batched images
	fmt.Println("Get images")
	images, answers := get_images()

	// Rescale images
	fmt.Println("Rescale images")
	rescaled_images := cn.rescale_images(images)

	// encode
	fmt.Println("Encode images")
	encoded_images := cn.encode_images(rescaled_images)

	// encrypt
	fmt.Println("Encrypt images")
	encrypted_images := cn.encrypt_images(encoded_images)

	// perform cryptonets
	predicted_images := cn.cryptonets(encrypted_images)

	// decrypt
	fmt.Println("Decrypt images")
	decrypted_images := cn.decrypt_fc(predicted_images)

	// decode
	fmt.Println("Decode images")
	decoded_images := cn.decode_fc(decrypted_images)

	// for i := 0; i < 1; i++ {
	// 	for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
	// 		for k := 0; k < PADDED_IMAGE_WIDTH; k++ {
	// 			data := float64(decoded_images[j][k][i])
	// 			fmt.Print(data, " ")
	// 		}
	// 		fmt.Println()
	// 	}
	// }

	// logSlots := cn.params.LogSlots()
	// values := make([]complex128, 1<<logSlots)

	// for i := 0; i < 1<<logSlots; i++ {
	// 	fmt.Println(rescaled_images[i][14][14])
	// 	values[i] = complex(float64(images[i][14][14]), 0)
	// }

	// for i := 0; i < 1<<logSlots; i++ {
	// 	fmt.Print(values[i], " ")
	// }
	// fmt.Println()

	// pt := cn.encode(values)
	// decoded_values := cn.decode(pt)

	// for i := 0; i < 1<<logSlots; i++ {
	// 	fmt.Print(real(decoded_values[i]), " ")
	// }
	// fmt.Println()

	// // Rescale images
	// fmt.Println("Rescale images")
	// rescaled_images := cn.rescale_images(images)

	// // encode
	// fmt.Println("Encode images")
	// encoded_images := cn.encode_images(rescaled_images)

	// // encrypt
	// fmt.Println("Encrypt images")
	// encrypted_images := cn.encrypt_images(encoded_images)

	// // perform cryptonets
	// // predicted_images := cn.cryptonets(encrypted_images)

	// // decrypt
	// fmt.Println("Decrypt images")
	// decrypted_images := cn.decrypt_images(encrypted_images)

	// // decode
	// fmt.Println("Decode images")
	// decoded_images := cn.decode_images(decrypted_images)

	// decoded_images
	fmt.Println(" ========================================= ")
	fmt.Println(" ============ Final Output =============== ")
	fmt.Println(" ========================================= ")
	errs := 0
	for i := 0; i < SLOTS; i++ {
		index := 0
		val := 0.

		for j := 0; j < FC_LAYERS; j++ {
			data := float64(decoded_images[j][i])
			if data > val {
				index = j
			}
		}

		if answers[i] != index {
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

	// for i := 0; i < 1; i++ {
	// 	for j := 0; j < POOL_LAYERS; j++ {
	// 		data := float64(decoded_images[j][i])
	// 		fmt.Print(data, " ")
	// 	}
	// }

	// for i := 0; i < 1; i++ {
	// 	for j := 0; j < FC_LAYERS; j++ {
	// 		data := float64(decoded_images[j][i])
	// 		fmt.Print(data, " ")
	// 	}
	// }

	// for i := 0; i < 1; i++ {
	// 	for j := 0; j < CONV_MAP; j++ {
	// 		for k := 0; k < CONV_SIZE; k++ {
	// 			for l := 0; l < CONV_SIZE; l++ {
	// 				data := float64(decoded_images[j][k][l][i])
	// 				fmt.Print(data, " ")
	// 			}
	// 			fmt.Println()
	// 		}
	// 		fmt.Println()
	// 	}
	// }

	// for i := 0; i < 1; i++ {
	// 	for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
	// 		for k := 0; k < PADDED_IMAGE_WIDTH; k++ {
	// 			if -1 < rescaled_images[i][j][k] && rescaled_images[i][j][k] < 1 {
	// 				fmt.Print(" ", " ")
	// 			} else {
	// 				fmt.Print(rescaled_images[i][j][k], " ")
	// 			}
	// 		}
	// 		fmt.Println()
	// 	}
	// }

	// fmt.Println()
	// fmt.Println()
	// fmt.Println()

	// for i := 0; i < 1; i++ {
	// 	for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
	// 		for k := 0; k < PADDED_IMAGE_WIDTH; k++ {
	// 			if -1 < decoded_images[j][k][i] && decoded_images[j][k][i] < 1 {
	// 				fmt.Print(" ", " ")
	// 			} else {
	// 				fmt.Print("*", " ")
	// 			}
	// 		}
	// 		fmt.Println()
	// 	}
	// }
}
