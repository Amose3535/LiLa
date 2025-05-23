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

#region INIT

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
		var input_dim = from[0]
		var output_dim = from[1]
		weights = Matrix.random(output_dim, input_dim, [-1.0, 1.0])
		bias = Matrix.generate_matrix(output_dim, 1, 0.0)
		return
	elif from is Array and from.size() == 3 and typeof(from[0]) == TYPE_INT and typeof(from[1]) == TYPE_INT and typeof(from[2]) == TYPE_ARRAY:
		var input_dim = from[0]
		var output_dim = from[1]
		weights = Matrix.random(output_dim, input_dim, from[2])
		bias = Matrix.generate_matrix(output_dim, 1, 0.0)
		return 
	
	push_error("DenseLayer init error: Unsupported init format.")
#endregion


#region OPERATIONS

## Performs the forward pass of the layer
func forward(input: Matrix) -> Matrix:
	#print("weights: ", weights.rows, "x", weights.columns)
	#print("input: ", input.rows, "x", input.columns)
	#print("bias: ", bias.rows, "x", bias.columns)
	var z = weights.row_column_mult(input).add(bias)
	return z.map(func(x): return 1.0 / (1.0 + exp(-x)))  # sigmoid


## Performs a single backpropagation step to adjust weights and bias
## using mean squared error and sigmoid activation derivative.
func train(input: Matrix, target: Matrix, learning_rate: float = 0.1) -> void:
	if !input.is_valid() or !target.is_valid():
		push_error("train(): Invalid input or target.")
		return
	
	# --- FORWARD PASS ---
	#Matrix.debug_shape(weights,"Weights")
	#Matrix.debug_shape(input,"Input")
	#Matrix.debug_shape(bias,"Bias")
	var z = weights.row_column_mult(input).add(bias)
	#Matrix.debug_shape(z,"Z")
	var output = z.map(func(x): return 1.0 / (1.0 + exp(-x)))  # sigmoid(x)
	#Matrix.debug_shape(output,"Output")
	
	# --- CALCULATE ERROR ---
	var error = Matrix.new(target).subtract(output) # copy the target matrix onto a new one and subtract from ti theoutput. That's the error
	
	# --- DERIVATIVE OF SIGMOID ---
	var gradient = output.map(func(x): return x * (1.0 - x))
	
	# --- delta = error * sigmoid'(z) ---
	var delta = error.hadamard_mult(gradient)
	
	# --- gradient descent step ---
	var delta_weights = delta.row_column_mult(Matrix.new(input).transpose()).multiply_by(learning_rate)
	var delta_bias = delta.multiply_by(learning_rate)
	
	# --- UPDATE weights and bias ---
	weights.add(delta_weights)
	bias.add(delta_bias)


## Returns a Dictionary from the current DenseLayer
func to_dict() -> Dictionary:
	if !weights.is_valid() or !bias.is_valid():
		push_error("to_dict(): Weights and/or Bias matrices are invalid.")
		return {}
	
	return {
		"weights": weights.to_dict(),
		"bias": bias.to_dict()
	}


## Returns a Dense Layer from a provided Dictionary
static func from_dict(data: Dictionary) -> DenseLayer:
	if not data.has("weights") or not data.has("bias"):
		push_error("DenseLayer.from_dict(): Missing keys.")
		return DenseLayer.new()
	
	var layer = DenseLayer.new()
	layer.weights = Matrix.from_dict(data["weights"])
	layer.bias = Matrix.from_dict(data["bias"])
	return layer

#endregion
