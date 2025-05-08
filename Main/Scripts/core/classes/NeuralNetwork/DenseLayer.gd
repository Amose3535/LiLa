class_name DenseLayer
extends Object


#region Internal Variables
var weights: Matrix
var bias: Matrix

var input_size: int: get = get_input_size
var output_size: int: get = get_output_size

func get_input_size() -> int:
	return weights.columns if weights.is_valid() else 0

func get_output_size() -> int:
	return weights.rows if weights.is_valid() else 0
#endregion

func _init(from: Variant = null) -> void:
	if from == null:
		weights = Matrix.generate_matrix(1, 1, 0.0)
		bias = Matrix.generate_matrix(1, 1, 0.0)
		return
	
	# If from is a DenseLayer → clone its weights and bias
	if from is DenseLayer:
		weights = Matrix.new(from.weights)
		bias = Matrix.new(from.bias)
		return
	
	# If from is a Dictionary with weights and bias
	if from is Dictionary and from.has("weights") and from.has("bias"):
		if from["weights"] is Matrix and from["bias"] is Matrix:
			weights = from["weights"]
			bias = from["bias"]
			return
		else:
			push_error("DenseLayer init error: weights and bias must be Matrix objects.")
			return
	
	# If from is an Array like [input_size : int, output_size : int]
	if from is Array and from.size() == 2 and typeof(from[0]) == TYPE_INT and typeof(from[1]) == TYPE_INT:
		var input_size = from[0]
		var output_size = from[1]
		weights = Matrix.random(output_size, input_size, [-1.0, 1.0])
		bias = Matrix.generate_matrix(output_size, 1, 0.0)
		return
	
	push_error("DenseLayer init error: Unsupported init format.")


## Performs the forward pass of the layer
func forward(input: Matrix) -> Matrix:
	print("weights: ", weights.rows, "x", weights.columns)
	print("input: ", input.rows, "x", input.columns)
	print("bias: ", bias.rows, "x", bias.columns)
	var z = weights.row_column_mult(input).add(bias)
	return z.map(func(x): return 1.0 / (1.0 + exp(-x)))  # sigmoid


## Performs a single backpropagation step to adjust weights and bias
## using mean squared error and sigmoid activation derivative.
func train(input: Matrix, target: Matrix, learning_rate: float = 0.1) -> void:
	if !input.is_valid() or !target.is_valid():
		push_error("train(): Invalid input or target.")
		return
	
	# --- FORWARD PASS ---
	var z = weights.row_column_mult(input).add(bias)
	var output = z.map(func(x): return 1.0 / (1.0 + exp(-x)))  # sigmoid(x)
	
	# --- CALCULATE ERROR ---
	var error = Matrix.new(target).subtract(output) # copy the target matrix onto a new one and subtract from ti theoutput. That's the error
	
	# --- DERIVATIVE OF SIGMOID ---
	var gradient = output.map(func(x): return x * (1.0 - x))
	
	# --- delta = error * sigmoid'(z) ---
	var delta = error.hadamard_mult(gradient)
	
	# --- gradient descent step ---
	var delta_weights = delta.row_column_mult(input.transpose()).multiply_by(learning_rate)
	var delta_bias = delta.multiply_by(learning_rate)
	
	# --- UPDATE weights and bias ---
	weights.add(delta_weights)
	bias.add(delta_bias)
