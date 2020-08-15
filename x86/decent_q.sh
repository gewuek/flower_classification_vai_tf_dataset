vai_q_tensorflow quantize \
	--input_frozen_graph ./frozen_graph.pb \
	--input_nodes conv2d_input \
	--input_shapes ?,128,128,3 \
	--output_nodes dense_1/Softmax \
	--input_fn flower_classification_input_fn.calib_input \
	--method 1 \
	--gpu 0 \
	--calib_iter 10 \
	--output_dir ./quantize_results
