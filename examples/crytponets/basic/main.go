package main 

import (
	"fmt" 
	"github.com/tuneinsight/lattigo/v4/rlwe"
	"github.com/tuneinsight/lattigo/v4/bfv"
)

func main() {
	// Creating encryption parameters from a default params with N=8192
	paramsDef := bfv.PN13QP218
	paramsDef.T = 65537
	params, err := bfv.NewParametersFromLiteral(paramsDef)
	if err != nil {
		panic(err)
	}
	encoder := bfv.NewEncoder(params)
	kgen := bfv.NewKeyGenerator(params)
	sk, _ := kgen.GenKeyPair()
	decryptor := bfv.NewDecryptor(params, sk)
	// encryptor_pk := bfv.NewEncryptor(params, pk)
	encryptor_sk := bfv.NewEncryptor(params, sk)
	evaluator := bfv.NewEvaluator(params, rlwe.EvaluationKey{})

	v := make([]uint64, 3)
	for i := 0; i < 3; i++ {
		v[i] = uint64(i)
	}

	pt := bfv.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(v, pt)

	ct := encryptor_sk.EncryptNew(pt)

	// sum all slots 
	evaluator.Add(ct, ct, ct) // add ct(1) with ct(2), and store in ct

	// decrypt
	plaintext := bfv.NewPlaintext(params, ct.Level())
	decryptor.Decrypt(ct, plaintext)

	// decode
	coeffsOut := make([]uint64, params.N())
	encoder.Decode(plaintext, coeffsOut)

	fmt.Println("coeffsOut:", coeffsOut)
}