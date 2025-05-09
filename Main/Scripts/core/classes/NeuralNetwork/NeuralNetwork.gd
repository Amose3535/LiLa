class_name NeuralNetwork
extends Object

var layers: Array[DenseLayer] = []

#region INIT

## _init function can make a NeuralNetwork out of many  different inputs.
##
## Supported from types:
## - null: Makes an empty NeuralNetwork
## - Array: Each element of the array must be of type Array of itself since it'll get used as initializer of DenseLayer class
##   Example: [code]NeuralNetwork.new([[2, 5, [-1,1],[5,1])[/code] will produce a NN with 2 Layers that gets 2 inputs and spits 1 output where the weights of the first layer are initialized randomòy btw -1 and 1
## - String (EXPERIMENTAL): This should basically do the same as Array 'from' type but in a more compact way. Needs a FIX
##
func _init(from: Variant = null) -> void:
	layers = []
	
	# Nessuna configurazione fornita: rete vuota, dovrai aggiungere i layer a mano
	if from == null:
		return
	
	# Se è un array trattare
	elif from is Array:#         2in 5out  5in 1out       2in 5out w/v btw-1&1 5in 1out 
		# Supporta sintassi tipo: [[2, 5], [5, 1]] oppure [[2, 5, [-1, 1]], [5, 1]]
		for layer_desc in from:
			if layer_desc is Array:
				layers.append(DenseLayer.new(layer_desc))
			else:
				push_error("Invalid layer format in NeuralNetwork._init. Expected Array per layer.")
	
	elif from is String:
		# Esempio stringa: "2>5>1" o "2>5[-1,1]>1"
		var pattern = from.strip_edges().split(">")
		var parsed = []
		for i in range(pattern.size() - 1 ):
			var current = pattern[i]
			var next = pattern[i + 1]
	
			var input_size = int(current)
			var output_desc = next
	
			var output_size = 0
			var interval = null
	
			# Supporta formato tipo 5[-2,2]
			if "[" in output_desc and "]" in output_desc:
				var num_part = output_desc.split("[")[0]
				var range_part = output_desc.split("[")[1].replace("]", "")
				output_size = int(num_part)
				var interval_array = range_part.split(",")
				interval = [float(interval_array[0]), float(interval_array[1])]
			else:
				output_size = int(output_desc)
	
			if interval != null:
				parsed.append([input_size, output_size, interval])
			else:
				parsed.append([input_size, output_size])
	
		# Ora parsed contiene l'array di array da passare a DenseLayer
		for layer in parsed:
			layers.append(DenseLayer.new(layer))
	else:
		push_error("Unsupported init format for NeuralNetwork.")
#endregion


#region OPERATIONS
func add_layer(layer: DenseLayer) -> void:
	layers.append(layer)


## Computes the forward function iteratively for each layer and return the result
func forward(input: Matrix) -> Matrix:
	var current_output = input
	for layer in layers:
		current_output = layer.forward(current_output)
	return current_output


## Train the entire neural network on a single input-target pair.
func train(input: Matrix, target: Matrix, learning_rate: float = 0.1) -> void:
	# === FORWARD PASS ===
	var activations: Array[Matrix] = []   # salviamo gli output di ogni layer
	var current_output = input
	activations.append(current_output)    # salviamo anche l’input come "output" del layer -1
	
	for layer in layers:
		current_output = layer.forward(current_output)
		activations.append(current_output)
	
	# === BACKWARD PASS ===
	var error = Matrix.new(target).subtract(current_output)  # errore finale: target - output
	var delta = error.hadamard_mult(activations.back().map(
		func(x): return x * (1.0 - x)  # derivata della sigmoid
	))
	
	# Loop a ritroso nei layer
	for i in range(layers.size() - 1, -1, -1):
		var layer = layers[i]
		var prev_activation = activations[i]
	
		# Aggiorna i pesi e bias del layer corrente
		var weight_update = delta.row_column_mult(Matrix.new(prev_activation).transpose()).multiply_by(learning_rate)
		var bias_update = delta.multiply_by(learning_rate)
	
		layer.weights.add(weight_update)
		layer.bias.add(bias_update)
	
		# Se non siamo al primo layer, calcola il delta per il layer precedente
		if i > 0:
			var prev_output = activations[i]
			var deriv = prev_output.map(func(x): return x * (1.0 - x))  # sigmoid'
			delta = Matrix.new(layer.weights).transpose().row_column_mult(delta).hadamard_mult(deriv)


## Computes the loss of the Network function (smaller = better)
func compute_loss(inputs: Array, targets: Array) -> float:
	if inputs.size() != targets.size():
		push_error("compute_loss(): Inputs and targets size mismatch.")
		return -1.0
	
	var total_loss := 0.0
	for i in range(inputs.size()):
		var prediction = forward(inputs[i])
		var target = targets[i]
		var error = target.subtract(prediction)
		var squared_error = error.map(func(x): return x * x)
		total_loss += squared_error.elements[0][0]  # only 1 output assumed
	
	return total_loss / inputs.size()


## Returns a Dictionary from the current NeuralNetwork
func to_dict() -> Dictionary:
	var serialized_layers = []
	for layer in layers:
		if (layer.weights.is_valid() and layer.bias.is_valid()):
			serialized_layers.append(layer.to_dict())
		else:
			push_error("to_dict(): Unable to make serialized layer from invalid layer weight or layer bias")
	return {
		"layers": serialized_layers
	}


## Returns a NeuralNetwork from a data Dictionary
static func from_dict(data: Dictionary) -> NeuralNetwork:
	if not data.has("layers"):
		push_error("NeuralNetwork.from_dict(): Missing 'layers' key.")
		return NeuralNetwork.new()
	var net = NeuralNetwork.new()
	for layer_data in data["layers"]:
		net.add_layer(DenseLayer.from_dict(layer_data))
	return net

#endregion

#region UTILS
#endregion
