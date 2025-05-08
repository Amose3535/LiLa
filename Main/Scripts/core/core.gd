extends Node

const EPSILON = 0.00001

# Called when the node enters the scene tree for the first time.
func _ready():
	if OS.has_feature("debug"):
		print("Calling core.test_matrix_class()")
		core.test_matrix_class()


# Called every frame. 'delta' is the elapsed time since the previous frame.
@warning_ignore("unused_parameter")
func _process(delta):
	pass


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
#endregion
