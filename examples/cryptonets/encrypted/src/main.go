package main

import (
	"encrypted/src/mnist"
	"encrypted/src/model"
	"fmt"

	"github.com/tuneinsight/lattigo/v4/bfv"
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

func rescale_images(images [][][]float64) [][][]uint64 {
	rescaled_images := make([][][]uint64, SLOTS)
	scale := 16.
	for i := 0; i < SLOTS; i++ {
		rescaled_image := make([][]uint64, PADDED_IMAGE_WIDTH)
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			rescaled_col := make([]uint64, PADDED_IMAGE_WIDTH)
			for k := 0; k < PADDED_IMAGE_WIDTH; k++ {
				rescaled_col[k] = uint64(images[i][j][k] * scale)
			}
			rescaled_image[j] = rescaled_col
		}
		rescaled_images[i] = rescaled_image
	}

	return rescaled_images
}

// Get encryption parameters from a default params with N=8192
func params() bfv.Parameters {
	paramsDef := bfv.PN13QP218
	paramsDef.T = 65537
	params, err := bfv.NewParametersFromLiteral(paramsDef)
	if err != nil {
		panic(err)
	}
	return params
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

func (cn CryptoNets) encode_images(images [][][]uint64) [][]*rlwe.Plaintext {
	encoded_images := make([][]*rlwe.Plaintext, PADDED_IMAGE_WIDTH)
	for i := 0; i < PADDED_IMAGE_WIDTH; i++ {
		encoded_col := make([]*rlwe.Plaintext, PADDED_IMAGE_WIDTH)
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			pixel := make([]uint64, SLOTS)
			for k := 0; k < SLOTS; k++ {
				pixel[k] = images[k][i][j]
			}
			encoded_pixel := cn.encode(pixel)
			encoded_col[j] = encoded_pixel
		}
		encoded_images[i] = encoded_col
	}
	return encoded_images
}

func (cn CryptoNets) encrypt(image *rlwe.Plaintext) *rlwe.Ciphertext {
	ct := bfv.NewCiphertext(cn.params, 1, cn.params.MaxLevel())
	cn.encryptor.Encrypt(image, ct)
	return ct
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
	pt := bfv.NewPlaintext(cn.params, encrypted.Level())
	cn.decryptor.Decrypt(encrypted, pt)
	return pt
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

func (cn CryptoNets) decode(decrypted *rlwe.Plaintext) []uint64 {
	// decode
	decoded := make([]uint64, SLOTS)
	cn.encoder.Decode(decrypted, decoded)
	return decoded
}

func (cn CryptoNets) decode_images(decrypted_images [][]*rlwe.Plaintext) [][][]uint64 {
	// decode
	decoded_images := make([][][]uint64, PADDED_IMAGE_WIDTH)
	for i := 0; i < PADDED_IMAGE_WIDTH; i++ {
		decoded_col := make([][]uint64, PADDED_IMAGE_WIDTH)
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			decoded_image := cn.decode(decrypted_images[i][j])
			decoded_col[j] = decoded_image
		}
		decoded_images[i] = decoded_col
	}

	// reformat images
	reformat_images := make([][][]uint64, SLOTS)
	for i := 0; i < SLOTS; i++ {
		image := make([][]uint64, PADDED_IMAGE_WIDTH)
		for j := 0; j < PADDED_IMAGE_WIDTH; j++ {
			image_col := make([]uint64, PADDED_IMAGE_WIDTH)
			for k := 0; k < PADDED_IMAGE_WIDTH; k++ {
				image_col[k] = decoded_images[j][k][i]
			}
			image[j] = image_col
		}
		reformat_images[i] = image
	}
	return reformat_images
}

func (cn CryptoNets) decode_conv(decrypted_images [][][]*rlwe.Plaintext) [][][][]uint64 {
	// decode
	decoded_images := make([][][][]uint64, CONV_MAP)
	for i := 0; i < CONV_MAP; i++ {
		decoded_row := make([][][]uint64, CONV_SIZE)
		for j := 0; j < CONV_SIZE; j++ {
			decoded_col := make([][]uint64, CONV_SIZE)
			for k := 0; k < CONV_SIZE; k++ {
				decoded_col[k] = cn.decode(decrypted_images[i][j][k])
			}
			decoded_row[j] = decoded_col
		}
		decoded_images[i] = decoded_row
	}

	// reformat images
	reformat_images := make([][][][]uint64, SLOTS)
	for i := 0; i < SLOTS; i++ {
		reformat_image := make([][][]uint64, CONV_MAP)
		for j := 0; j < CONV_MAP; j++ {
			reformat_row := make([][]uint64, CONV_SIZE)
			for k := 0; k < CONV_SIZE; k++ {
				reformat_col := make([]uint64, CONV_SIZE)
				for l := 0; l < CONV_SIZE; l++ {
					reformat_col[l] = decoded_images[j][k][l][i]
				}
				reformat_row[k] = reformat_col
			}
			reformat_image[j] = reformat_row
		}
		reformat_images[i] = reformat_image
	}
	return reformat_images
}

func (cn CryptoNets) decode_pool(encoded_image []*rlwe.Plaintext) [][]uint64 {
	decoded_image := make([][]uint64, POOL_LAYERS)
	for i := 0; i < POOL_LAYERS; i++ {
		decoded_image[i] = cn.decode(encoded_image[i])
	}

	// reformat images
	reformat_images := make([][]uint64, SLOTS)
	for i := 0; i < SLOTS; i++ {
		reformat_image := make([]uint64, POOL_LAYERS)
		for j := 0; j < POOL_LAYERS; j++ {
			reformat_image[j] = decoded_image[j][i]
		}
		reformat_images[i] = reformat_image
	}

	return reformat_images
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
					// moduli, but we're only wrapping around the first one
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

func (cn CryptoNets) convolution_layer(encrypted_images [][]*rlwe.Ciphertext) [][][]*rlwe.Ciphertext {
	// Prepare model weights / bias
	conv_kernels, conv_bias := cn.prepare_conv()

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
						window[i*5+j] = encrypted_images[(i + offset_y)][(j + offset_x)]
					}
				}
				windows[stride_y*13+stride_x] = window

				// dot product
				product := make([]*rlwe.Ciphertext, 25)

				// kernel is already in column major form
				kernel := conv_kernels[o]

				for i := 0; i < 25; i++ {
					ct := bfv.NewCiphertext(cn.params, window[0].Degree(), window[0].Level())
					cn.evaluator.Mul(window[i], kernel[i], ct)
					product[i] = ct
				}

				// sum all products together
				sum := product[0]
				for i := 1; i < 25; i++ {
					cn.evaluator.Add(sum, product[i], sum)
				}

				// add bias
				bias := conv_bias[o]
				cn.evaluator.Add(sum, bias, sum)

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
				new_ct := bfv.NewCiphertext(cn.params, 2, ct.Level())
				cn.evaluator.Mul(ct, ct, new_ct)
				squared_col[k] = new_ct
			}
			squared_row[j] = squared_col
		}
		squared[i] = squared_row
	}
	return squared
}

func (cn CryptoNets) prepare_pool_layer() ([][]*rlwe.Plaintext, []*rlwe.Plaintext) {
	// rescale kernels
	// scale: 32
	scale := 32.
	pool_layers := make([][]float64, POOL_LAYERS)
	for i := 0; i < POOL_LAYERS; i++ {
		pool_layer := make([]float64, CONVOLUTION_SIZE)
		for j := 0; j < CONVOLUTION_SIZE; j++ {
			// transpose weights
			val := model.Weights_1[i+j*POOL_LAYERS]
			val *= scale
			if val < 0 {
				val += float64(cn.params.T())
			}
			pool_layer[j] = val
		}
		pool_layers[i] = pool_layer
	}

	// encode
	encoded_kernels := make([][]*rlwe.Plaintext, POOL_LAYERS)
	for i := 0; i < POOL_LAYERS; i++ {
		encoded_kernel := make([]*rlwe.Plaintext, CONVOLUTION_SIZE)
		for j := 0; j < CONVOLUTION_SIZE; j++ {
			val := pool_layers[i][j]
			kernel := make([]uint64, 8192)
			for k := 0; k < 8192; k++ {
				kernel[k] = uint64(val)
			}
			encoded_kernel[j] = cn.encode(kernel)
		}
		encoded_kernels[i] = encoded_kernel
	}

	// encode conv bias
	encoded_bias := make([]*rlwe.Plaintext, POOL_LAYERS)
	for i := 0; i < POOL_LAYERS; i++ {
		val := model.PoolBias[i]
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

func (cn CryptoNets) pool_layer(sq_layer_1 [][][]*rlwe.Ciphertext) []*rlwe.Ciphertext {
	pool_kernels, pool_bias := cn.prepare_pool_layer()

	outputs := make([]*rlwe.Ciphertext, POOL_LAYERS)
	for o := 0; o < POOL_LAYERS; o++ {
		sum := bfv.NewCiphertext(cn.params, 2, sq_layer_1[0][0][0].Level())
		index := 0
		for i := 0; i < CONV_MAP; i++ {
			for j := 0; j < CONV_SIZE; j++ {
				for k := 0; k < CONV_SIZE; k++ {
					ct := sq_layer_1[i][j][k]
					res := bfv.NewCiphertext(cn.params, ct.Degree(), ct.Level())
					cn.evaluator.Mul(ct, pool_kernels[o][index], res)
					cn.evaluator.Add(sum, res, sum)
					index++
				}
			}
		}

		// add bias
		bias := pool_bias[o]
		cn.evaluator.Add(sum, bias, sum)

		outputs[o] = sum
	}
	return outputs
}

func (cn CryptoNets) cryptonets(encrypted_images [][]*rlwe.Ciphertext) []*rlwe.Ciphertext {
	fmt.Println("Cryptonets!")
	// Inference layers (see Table 1 of `Paper source`)

	// 1. Convolution layer
	conv_layer := cn.convolution_layer(encrypted_images)

	// 2. Square activation layer
	sq_layer_1 := cn.square_layer_1(conv_layer)

	// 3. Pooling layer
	pool_layer := cn.pool_layer(sq_layer_1)

	return pool_layer
}

func main() {
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

	// Get batched images
	images, _ := get_images()

	// Rescale images
	rescaled_images := rescale_images(images)

	// encode
	encoded_images := cn.encode_images(rescaled_images)

	// encrypt
	encrypted_images := cn.encrypt_images(encoded_images)

	// perform cryptonets
	predicted_images := cn.cryptonets(encrypted_images)

	// decrypt
	decrypted_images := cn.decrypt_pool(predicted_images)

	// decode
	decoded_images := cn.decode_pool(decrypted_images)

	// decoded_images
	fmt.Println(" ========================================= ")
	fmt.Println(" ============ Final Output =============== ")
	fmt.Println(" ========================================= ")
	for i := 0; i < 1; i++ {
		for j := 0; j < POOL_LAYERS; j++ {
			data := float64(int16(decoded_images[i][j]))
			data /= 32. // rescale after convolution
			data /= 32. // rescale after square
			fmt.Print(data, " ")
		}

	}
}
