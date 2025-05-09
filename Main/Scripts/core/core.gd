extends Node

const EPSILON = 0.00001
const xor_nn_path : String = "user://XOR_neural_network.json"

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	if OS.has_feature("debug"):
		#core.test_matrix_class()
		#core.test_dense_layer_training_test()
		#core.test_neural_network_xor(6,0.25,100000)
		return


## Saves a dictionary as JSON file to 'path'
func save_dict(dict: Dictionary, path: String) -> bool:
	var file := FileAccess.open(path, FileAccess.WRITE)
	if file != null:
		file.store_string(JSON.stringify(dict, "", false, true))
		file.close()
		return true
	else:
		push_error("Impossibile aprire il file per la scrittura: " + path)
		return false


## Loads dictionary from JSON file
func load_dict(path: String) -> Dictionary:
	if !FileAccess.file_exists(path):
		push_error("Il file non esiste: " + path)
		return {}
	
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		push_error("Impossibile aprire il file per la lettura: " + path)
		return {}
	
	var content := file.get_as_text()
	file.close()
	
	var parsed = JSON.parse_string(content)
	if typeof(parsed) != TYPE_DICTIONARY:
		push_error("Contenuto JSON non valido in: " + path)
		return {}
	
	return parsed


#region TESTING functions
func test_matrix_class() -> void:
	print("----- TEST STARTED -----")	
	# Test 1: default init
	var m1 = Matrix.new()
	print("Default init (should be 1x1 zero):")
	Matrix.printm(m1)
	
	# Test 2: array init
	var m2 = Matrix.new([[1, 2], [3, 4]])
	print("Array init (2x2):")
	Matrix.printm(m2)
	
	# Test 3: matrix copy
	var m3 = Matrix.new(m2)
	print("Copied matrix (should match m2):")
	Matrix.printm(m3)
	
	# Test 4: string init "3*2|7"
	var m4 = Matrix.new("3*2|7")
	print("String init '3*2|7':")
	Matrix.printm(m4)
	
	# Test 5: identity matrix
	var m5 = Matrix.identity(3)
	print("Identity 3x3:")
	Matrix.printm(m5)
	
	# Test 6: multiply_by (in-place)
	print("m2 before multiply_by(3):")
	Matrix.printm(m2)
	m2.multiply_by(3)
	print("m2 after multiply_by(3):")
	Matrix.printm(m2)
	
	# Test 7: row_column_mult valid
	var a = Matrix.new([[1, 2, 3], [4, 5, 6]])
	var b = Matrix.new([[7, 8], [9, 10], [11, 12]])
	var result = a.row_column_mult(b)
	print("row_column_mult A (2x3) * B (3x2):")
	Matrix.printm(result)
	
	# Test 8: row_column_mult invalid
	var invalid = Matrix.new([[1, 2]])
	var invalid_result = invalid.row_column_mult(m5)  # Should trigger error and return 1x1 zero
	print("Invalid row_column_mult:")
	Matrix.printm(invalid_result)
	
	# Test 9: get_row & get_column
	var test = Matrix.new([[5, 6, 7], [8, 9, 10]])
	print("Row 1 of test:")
	print(test.get_row(1))  # [8, 9, 10]
	print("Column 0 of test:")
	print(test.get_column(0))  # [5, 8]
	
	# Test 10: garbage string init
	var garbage = Matrix.new("banana*pizza|lol")
	print("Garbage string init (should default 1x1):")
	Matrix.printm(garbage)
	
	print("----- TEST ENDED -----")

func test_dense_layer_training_test():
	# === SETUP ===
	print("\n=== INIT ===")
	var layer = DenseLayer.new([2, 1])  # 2 input → 1 output
	var input = Matrix.new([[1.0], [0.0]])  # colonna da 2 righe
	var target = Matrix.new([[1.0]])  # voglio che produca 1
	#Matrix.debug_shape(layer.weights,"Layer Weights")
	#Matrix.debug_shape(layer.bias,"Layer Bias")
	#Matrix.debug_shape(input,"Input")
	#Matrix.debug_shape(target,"Target")
	print("Layer Weights:")
	Matrix.printm(layer.weights)
	print("Layer Bias:")
	Matrix.printm(layer.bias)
	print("Input Matrix:")
	Matrix.printm(input)
	print("Input Target:")
	Matrix.printm(target)
	

	# === PRIMA DEL TRAINING ===
	print("\n=== BEFORE TRAINING ===")
	var output_before = layer.forward(input)
	print("Output prima: ")
	Matrix.printm(output_before)

	# === ALLENAMENTO ===
	for i in range(1001):
		layer.train(input, target, 0.1)
		if i % 100 == 0:
			var step_output = layer.forward(input)
			print("Epoch %d | Output: " % i)
			Matrix.printm(step_output)


	# === DOPO IL TRAINING ===
	print("\n=== AFTER TRAINING ===")
	var output_after = layer.forward(input)
	print("Output dopo: ")
	Matrix.printm(output_after)

	# Stampa finale
	print("\nPesi:\n")
	Matrix.printm(layer.weights)
	print("Bias:\n")
	Matrix.printm(layer.bias)

func test_neural_network_xor(layer_neuron_number : int = 5, learning_rate : float = 0.1, training_cycles: int = 5000):
	print("\n=== NEURAL NETWORK TEST STARTED ===")
	
	# Crea la rete neurale con 1 hidden layer e 1 output layer. Il primo prende 2 input e "layer neuron number"
	# output e l'output layer che prende "layer neuron number" input e sputa 1 solo output
	var net = NeuralNetwork.new([[2,layer_neuron_number,[-1.0,1.0]],[layer_neuron_number,1]])
	
	# XOR dataset
	var dataset : Dictionary = {
		Matrix.new([[0.0], [0.0]]) : Matrix.new([[0.0]]),
		Matrix.new([[0.0], [1.0]]) : Matrix.new([[1.0]]),
		Matrix.new([[1.0], [0.0]]) : Matrix.new([[1.0]]),
		Matrix.new([[1.0], [1.0]]) : Matrix.new([[0.0]])
	}
	
	var inputs = dataset.keys()
	
	var targets = dataset.values()
	
	# Allenamento
	for epoch in range(training_cycles+1):
		for i in range(inputs.size()):
			net.train(inputs[i], targets[i], learning_rate)
	
		if epoch % 500 == 0:
			print("Epoch %d" % epoch)
			for i in range(inputs.size()):
				var out = net.forward(inputs[i])
				print("Input: ", inputs[i].mat_to_str(), "→ Output: ", out.mat_to_str())
	
	# Risultato finale
	print("\n=== NEURAL NETWORK TEST ENDED ===")
	for i in range(inputs.size()):
		var output = net.forward(inputs[i])
		print("Input: ", inputs[i].mat_to_str(), "→ Output: ", output.mat_to_str())
	
	print("Attempting to save XOR NeuralNetwork to file...")
	if save_dict(net.to_dict(),xor_nn_path):
		print("Successfully saved XOR neural network to: %s"%[xor_nn_path])
	else:
		print("Unable to save XOR neural network to: %s"%[xor_nn_path])


#endregion
