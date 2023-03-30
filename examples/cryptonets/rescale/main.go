package main

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v4/ckks"
	"github.com/tuneinsight/lattigo/v4/rlwe"
)

// Get encryption parameters from a default params with N=8192
func params() ckks.Parameters {
	paramsDef := ckks.PN14QP438
	params, err := ckks.NewParametersFromLiteral(paramsDef)
	if err != nil {
		panic(err)
	}
	return params
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

	values := make([]complex128, 8192)
	for i := 0; i < 8192; i++ {
		values[i] = complex(float64(i), 0)
		fmt.Print(values[i], " ")
	}
	fmt.Println()
	fmt.Println()

	pt := encoder.EncodeNew(values, params.MaxLevel(), params.DefaultScale(), params.LogSlots())
	ct := encryptor_sk.EncryptNew(pt)
	constant := params.Q()[0]
	evaluator.MultByConst(ct, constant, ct)
	fmt.Println("const:", constant)
	decrypted_values := decryptor.DecryptNew(ct)
	decoded_values := encoder.Decode(decrypted_values, params.LogSlots())

	decoded_real := make([]float64, 10)
	for i := 0; i < 10; i++ {
		decoded_real[i] = real(decoded_values[i])
		fmt.Print(decoded_real[i], " ")
	}
	fmt.Println("")
	fmt.Println("")

	for i := 0; i < 9; i++ {
		constant := params.RingQ().SubRings[ct.Level()-i].Modulus
		fmt.Println("constant: ", constant)
		fmt.Println("default scale: ", params.DefaultScale())
		evaluator.MultByConst(ct, constant, ct)
		fmt.Println("before ct:scale: ", ct.Scale)
		ct.Scale = ct.Scale.Mul(rlwe.NewScale(constant))
		fmt.Println("after ct:scale: ", ct.Scale)

		fmt.Println("rescale")
	}

	decrypted_values = decryptor.DecryptNew(ct)
	decoded_real = make([]float64, 10)
	fmt.Println()
	fmt.Println()
	fmt.Println("before rescaling!!")
	fmt.Println(ct.Scale)
	for i := 0; i < 10; i++ {
		decoded_real[i] = real(decoded_values[i])
		fmt.Print(decoded_real[i], " ")
	}
	fmt.Println()
	fmt.Println("before rescaling!!")
	fmt.Println()
	fmt.Println()

	var new_ct *rlwe.Ciphertext = ct.CopyNew()
	if err := evaluator.Rescale(ct, params.DefaultScale(), new_ct); err != nil {
		panic("rescale failed")
	}

	fmt.Println("after rescale!!! ", new_ct.Scale)

	fmt.Println("params:max.Level()", params.MaxLevel())

	decrypted_values = decryptor.DecryptNew(ct)
	decoded_real = make([]float64, 10)
	for i := 0; i < 10; i++ {
		decoded_real[i] = real(decoded_values[i])
		// / 9223372036854 / 4
		fmt.Print(decoded_real[i], " ")
	}
	fmt.Println("")

	fmt.Println("default scale: ", params.DefaultScale().Float64())

	// ciphertext.Scale = ciphertext.Scale.Mul(rlwe.NewScale(constant))

	// if err := tc.evaluator.Rescale(ciphertext, tc.params.DefaultScale(), ciphertext); err != nil {
	// 	t.Fatal(err)
	// }

}
