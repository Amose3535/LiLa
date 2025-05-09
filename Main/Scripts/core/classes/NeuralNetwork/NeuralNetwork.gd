class_name NeuralNetwork
extends Object

var layers: Array[DenseLayer] = []

func _init():
	layers = []

func add_layer(layer: DenseLayer) -> void:
	layers.append(layer)

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

## Computes the loss of 
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
