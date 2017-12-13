Model	결과	Traning Step	Learning Rate	batch_size	optimizer	activation function	silence_percentage	unknown_percentage	time_shift_ms	sample_rate
M1	86.5%	15000/3000	0.01/0.001/0.0001	100	GradientDescentOptimizer	Relu	10	10	100	16000
M2	89.2%	15000/3000	0.01/0.001/0.0001	100	GradientDescentOptimizer	Relu				
M2	93.8%	8000/5000/3000	0.01/0.002/0.0001	100	RMSPropOptimizer	Relu				
										
										
										
										
M1	"Conv / s2
Conv dw / s1
Conv / s1
Avg Pool / s1"	"3 x 3 x 1 x 32 
3 x 3 x 32 x 32
1 x 1 x 32 x 64
"	"65 x 40 x 1
32 x 19 x 32
32 x 19 x 32
32 x 19 x 64"							
M2	"Conv / s2
Conv dw / s1
Conv / s1
Conv dw / s2
Conv / s1
Avg Pool / s1"	"3 x 3 x 1 x 32 
3 x 3 x 32 x 32
1 x 1 x 32 x 64
3 x 3 x 64 x 64
1 x 1 x 64 x 128"	"65 x 40 x 1
32 x 19 x 32
32 x 19 x 32
32 x 19 x 64
15 x 9 x 64
15 x 9 x 128
15 x 9 x 128
"							
