## A class used to do implement matrices and their most important operations
class_name Matrix
extends RefCounted # RefCounted>Object cause with huge amounts of matrices they auto-clean themselves when not used anymore

#region Internal variables

# Cannot nest Array types BUT Matrices default automatically to float. THEY CANNOT
# have anything else

## Elements is a grid-like structure containing the raw data of the matrix
var elements : Array[Array] = [[0.0]]

## Rows automatically update through the getter function and matches the number of rows in elements
var rows: int: get = get_rows

## Columns automatically update through the getter function and matches the number of columns in elements
var columns: int: get = get_columns

func get_rows() -> int: return elements.size()
func get_columns() -> int: return elements[0].size() if elements.size() > 0 else 0
#endregion


## the function called when the matrix is built through Matrix.new(...). 
# NOTE: when _init() is called there is an instance of this class => no need to instantiate a matrix, we can just edit the instanced one
# Accepts: Matrix, Array, String
# Returns: void
func _init(from : Variant = null) -> void:
	## from can either be:
	#- null (in that case instantiate a 1x1 matrix with a single 0.0 value)
	#- another matrix (in that case we want to instantiate a carbon copy of the original one)
	#- string (in that case either use format "r*c|fill_value")
	
	if from == null:
		# edit self (the instanced matrix) to a default 1*1|0.0 matrix
		edit_matrix(1,1,0.0)
		return
		
	elif from is Matrix:
		# check if from is a valid matrix (kinda redundant since in copy() there's already a check if it's valid but you're never too sure)
		if from.is_valid():
			copy(from)
		else:
			push_error("Unable to create a Matrix out of an invalid Matrix object.")
		return
		
	elif from is Array:
		if Matrix.array_is_valid_matrix(from):
			elements = []
			for row in from:
				elements.append(row.duplicate())  # per-row deep copy
		else:
			push_error("Matrix init error: Provided Array is not a valid matrix.")
		return
	
	# Format: "n*m|fill_value"
	elif from is String:
		if from.contains("|") and from.contains("*"):
			var data = from.split("|")
			if data.size() == 2:
				var dims = data[0].split("*")
				var fill_val = data[1].to_float()
				if dims.size() == 2 and dims[0].is_valid_float() and dims[1].is_valid_float():
					var r = int(dims[0])
					var c = int(dims[1])
					edit_matrix(r, c, fill_val)
				else:
					push_error("Matrix string init error: Invalid dimensions format.")
			else:
				push_error("Matrix string init error: Incorrect string split by '|'")
		else:
			push_error("Matrix string init error: Invalid _init parameters [source: %s]" % [str(from)])
		return
	
	else:
		push_error("Matrix init error: Unsupported type for initialization.")


#region CHECKS
## function to check if the instance of Matrix is valid
func is_valid() -> bool:
	# for the Matrix to be valid it MUSt be: 
	# NOT empty
	# The number of rows in elements MUST match rows var
	# The number of columns MUST remain consistent over each row
	
	if !Matrix.array_is_valid_matrix(elements):
		push_warning("Matrix's elements are unsuitable for a matrix")
		return false
	
	if elements.is_empty():
		push_warning("Matrix is empty.")
		return false
	
	if elements.size() != rows:
		push_warning("Matrix row count mismatch: expected %d, found %d" % [rows, elements.size()])
		return false
	
	var rlen: int = elements[0].size()
	if rlen != columns:
		push_warning("Matrix column count mismatch: expected %d, found %d" % [columns, rlen])
		return false
	
	for i in range(1, elements.size()):
		if elements[i].size() != rlen:
			push_warning("Row %d has inconsistent length (expected %d, found %d)" % [i, rlen, elements[i].size()])
			return false
	
	return true


## checks if the matrix is square
func is_square() -> bool:
	if !is_valid():
		push_error("is_square(): matrix is not valid.")
		return false
	return (rows == columns) # rows == columns internally equates to a boolean operator, no need to do any shit


## Returns true if all non-diagonal elements are zero.
func is_diagonal() -> bool:
	if !is_valid():
		push_warning("is_diagonal(): Matrix is not valid.")
		return false
	
	if !is_square():
		push_warning("is_diagonal(): Matrix must be square.")
		return false
	
	for r in range(rows):
		for c in range(columns):
			if r != c and abs(elements[r][c]) > core.EPSILON:
				return false
	
	return true


## Returns true if all values BELOW the diagonal are zero.
func is_upper_triangular() -> bool:
	if !is_valid():
		push_error("is_upper_triangular(): Matrix must be valid!")
		return false
	
	if !is_square():
		push_error("is_upper_triangular(): Matrix must be square!")
		return false
	
	for r in range(rows):
		for c in range(columns):
			if r > c and abs(elements[r][c]) > core.EPSILON:
				return false
	
	return true


## Returns true if all values BELOW the diagonal are zero.
func is_lower_triangular() -> bool:
	if !is_valid():
		push_error("is_lower_triangular(): Matrix must be valid!")
		return false
	
	if !is_square():
		push_error("is_lower_triangular(): Matrix must be square!")
		return false
	
	for r in range(rows):
		for c in range(columns):
			if r < c and abs(elements[r][c]) > core.EPSILON:
				return false
	
	return true


## Checls if all elements of the matrix are equal to zero (uses EPSILON check)
func is_zero() -> bool:
	if !is_valid():
		push_warning("is_zero(): Invalid matrix")
		return false
	
	for r in range(rows):
		for c in range(columns):
			if abs(elements[r][c]) > core.EPSILON:
	
				return false
	
	return true


## checks if two matrices are equal element by element
func equals(matrix : Matrix) -> bool:
	if !self.is_valid() or !matrix.is_valid():
		push_error("Comparison of matrices: One of the two matrices is invalid.")
		return false
	
	if ((matrix.rows != rows) || (matrix.columns != columns)):
		push_error("Comparison of matrices: Unable to equate two matrices with different sizes.")
		return false
	
	for r in range(rows):
		for c in range(columns):
			if (elements[r][c] - matrix.elements[r][c]) > core.EPSILON:
				return false
	
	return true


## checks if self is an identity matrix
func is_identity() -> bool:
	if !is_valid():
		push_error("is_identity(): Matrix is invalid")
		return false
	
	if !is_square():
		push_error("is_identity(): Matrix must be square to be identity.")
		return false
	
	return self.equals(Matrix.identity(rows)) # equates to true or false depending if matrix self equals the generated id matrix


## function to check if a provided array is suitable for matrix generation (Accessible from everywhere through Matrix.array_is_valid_matrix([...]) )
static func array_is_valid_matrix(array: Array) -> bool:
	if array.is_empty():
		return false
	
	var expected_cols = -1
	for row in array:
		if !(row is Array):
			return false
		if expected_cols == -1:
			expected_cols = row.size()
		elif row.size() != expected_cols:
			return false
		for val in row:
			if typeof(val) != TYPE_FLOAT and typeof(val) != TYPE_INT:
				return false
	
	return true
#endregion


#region COPY/GET
## function to clone another Matrix onto self
func copy(original_matrix : Matrix) -> void:
	## Before copying the matrix first make sure that it is a valid one
	if original_matrix.is_valid():
		elements = []
		for row in original_matrix.elements:
			elements.append(row.duplicate(true))
		
	else:
		push_warning("Unable to copy %s due to an invalid matrix, defaulting to '[[0.0]]'"%str(original_matrix))
		elements = [[0.0]]


## function to get the row in an array based on the index
func get_row(index: int) -> Array:
	if index >= 0 and index < rows:
		return elements[index]
	push_error("Row index out of bounds")
	return []


## function to get the column in an array based on the index
func get_column(index: int) -> Array:
	if index < 0 or index >= columns:
		return []
	
	var col : Array = []
	for row in elements:
		col.append(row[index])
	return col



## Returns a string representation of this matrix.
func mat_to_str() -> String:
	if !is_valid():
		push_error("mat_to_str(): Matrix is not valid.")
		return ""
	
	return str(elements)


## Returns an array containing the data of elements
func mat_to_arr() -> Array:
	if !is_valid():
		push_error("mat_to_vec(): Matrix is not valid.")
		return Array()
	
	return elements.duplicate(true)


## Returns a dictionary that is made up of the elements of the matrix. Useful for save serialization
func to_dict() -> Dictionary:
	if !is_valid():
		push_error("to_dict(): Unable to make dict from invalid Matrix")
		return {}
	
	return {
		"elements": self.elements
	}


## Returns a Matrix class from a dictionary. Useful for save deserialization
static func from_dict(dict: Dictionary) -> Matrix:
	if not dict.has("elements"):
		push_error("Matrix.from_dict(): Missing 'elements' key.")
		return Matrix.new()
	
	return Matrix.new(dict["elements"])


#endregion


#region GENERATION/EDITING

## Modify the instance of a matrix to match rows_number and columns_number and fill it with fill_value.
# THIS RETURNS ->VOID<-. ONLY USE WHEN MODIFYING AN INSTANCE. TO GET A MATRIX BACK USE generate_matrix() 
func edit_matrix(rows_number: int, columns_number: int, fill_value: float = 1.0) -> void:
	elements = []
	for i in range(rows_number):
		var row: Array = []
		for j in range(columns_number):
			row.append(fill_value)
		elements.append(row)


## Generate an instance of Matrix class with rows = rows_number, columns = columns_number, and filled with fill_value.
# (usually not used by itself but it is basically an alternative to Matrix.new("r*c|f_val"))
static func generate_matrix(rows_number: int, columns_number: int, fill_value: float = 1.0) -> Matrix:
	var return_matrix = Matrix.new("%s*%s|%s"%[rows_number,columns_number,fill_value])
	return return_matrix


## Make an identity matrix
# takes advantage of generate_matrix to make a square matrix where the diagonal is filled with ones
static func identity(size: int = 2) -> Matrix:
	var id_mat = generate_matrix(size, size, 0.0)
	for i in range(size):
		id_mat.elements[i][i] = 1.0
	return id_mat


## Make a matrix full of 1.0
# basically generates_matrix(...) but the fill_value is omitted (defaults to 1.0)
static func ones(rows_num: int, columns_num: int) -> Matrix:
	var o_mat = generate_matrix(rows_num,columns_num)
	return o_mat


## Make a matrix full of 0.0
# basically generates_matrix(...) but the fill_value is set to 0.0
static func zeros(rows_num: int, columns_num: int) -> Matrix:
	var z_mat = generate_matrix(rows_num,columns_num,0.0)
	return z_mat


## Generates a matrix filled with random values within a given interval [min, max].
static func random(rows_num: int, columns_num: int, interval: Array = [-1.0, 1.0]) -> Matrix:
	if interval.size() != 2:
		push_error("random(): interval must be an Array of two floats: [min, max].")
		return Matrix.zeros(rows_num, columns_num)
	
	var min_val = interval[0]
	var max_val = interval[1]
	var random_mat: Matrix = Matrix.zeros(rows_num, columns_num)
	
	for r in range(rows_num):
		for c in range(columns_num):
			random_mat.elements[r][c] = randf_range(min_val, max_val)
	
	return random_mat



## Calculates and returns the cofactor matrix of this matrix.
## Each element is: (-1)^(i+j) * determinant(minor(i,j))
func cofactor() -> Matrix:
	if !is_valid():
		push_error("cofactor_matrix(): Matrix is not valid.")
		return self
	
	if !is_square():
		push_error("cofactor_matrix(): Matrix must be square.")
		return self

	var cof = Matrix.generate_matrix(rows, columns, 0.0)

	for i in range(rows):
		for j in range(columns):
			var sign_value = (-1.0) ** (i + j)
			var minor = minor_matrix(i, j)
			cof.elements[i][j] = sign_value * minor.determinant()

	return cof


## Returns a copy of this matrix where each element is multiplied by its checkerboard sign.
## Sign is calculated as (-1)^(row + column)
func checker() -> Matrix:
	if !is_valid():
		push_error("checker(): Matrix is not valid.")
		return self
	
	var result = Matrix.generate_matrix(rows, columns, 0.0)
	
	for r in range(rows):
		for c in range(columns):
			var sign_value = (-1.0) ** (r + c)
			result.elements[r][c] = sign_value * elements[r][c]
	
	return result
#endregion


#region OPERATIONS

## function to multiply a scalar to each element of the matrix
func multiply_by(k : float = 1) -> Matrix:
	if is_valid():
		for row in range(rows):
			for cell in range(columns):
				elements[row][cell] = elements[row][cell]*k
	return self


## function used to execute dot multiplication between matrices
# dot mul works by getting the nth row of the first matrix and the nth row of the second matrix and outputting the sum of the 
# products of each element as the element of the matrix
func row_column_mult(operand : Matrix) -> Matrix:
	var result = Matrix.new()
	# If there is a mismatch between the rows of the first matrix and the columns of the second (basically the requirement to apply r/c mult)
	if self.columns != operand.rows:
		# push an error where we specify that the matrices have incompatible dimensions
		push_error("Matrix dot multiplication: incompatible dimensions.")
		return Matrix.generate_matrix(1, 1, 0.0)
	
	# If all goes well then we generate a result matrix with the same number of rows as the first one and the same number of col. of the second one
	result = Matrix.generate_matrix(rows, operand.columns)
	
	# and then we apply the r/c mult algorithm to the result Matrix where we move a cursor k along both the current row and column and add to the
	# sum the multiplication of the pointed element of the r-th row in the first matrix and the c-th column in the second matrix
	for r in range(rows):
		for c in range(operand.columns):
			var sum = 0.0
			for k in range(columns):
				sum += elements[r][k] * operand.elements[k][c]
			result.elements[r][c] = sum
	return result


## function to multiply element by element two matrices
func hadamard_mult(operand : Matrix) -> Matrix:
	if !self.is_valid() or !operand.is_valid():
		push_error("Multiplication of matrices: One of the two matrices is invalid.")
		return self
	
	if ((operand.rows != rows) || (operand.columns != columns)):
		push_error("Multiplication of matrices: Unable to multiply the two matrices element-by-element with different sizes.")
		return self
	
	for r in range(rows):
		for c in range(columns):
			elements[r][c] *= operand.elements[r][c]
	return self


## function used to pass from a nxm matrix to a mxn matrix with rotation of the elements
func transpose() -> Matrix:
	# Create a new matrix that's a copy of the instance
	var old_mat : Matrix = Matrix.new(self)
	# Edit the original matrix to switch the number of columns and rows with a default value
	self.edit_matrix(self.columns,self.rows,0.0)
	
	# populate the rows of the instance by cloning the columns of the original matrix 
	for i in range(old_mat.columns):
		var col = old_mat.get_column(i)
		self.elements[i] = col
	
	return self


## function used to add two matrices together
func add(matrix : Matrix) -> Matrix:
	if !self.is_valid() or !matrix.is_valid():
		push_error("Sum of matrices: One of the two matrices is invalid.")
		return self
	
	if ((matrix.rows != rows) || (matrix.columns != columns)):
		push_error("Sum of matrices: Unable to sum two matrices with different sizes.")
		return self
	
	for r in range(rows):
		for c in range(columns):
			elements[r][c] += matrix.elements[r][c]
	return self


## function used to subtract two matrices together (same as add but simply with a - operator)
func subtract(matrix : Matrix) -> Matrix:
	if !self.is_valid() or !matrix.is_valid():
		push_error("Subtraction of matrices: One of the two matrices is invalid.")
		return self
	
	if ((matrix.rows != rows) || (matrix.columns != columns)):
		push_error("Subtraction of matrices: Unable to subtract two matrices with different sizes.")
		return self
	
	for r in range(rows):
		for c in range(columns):
			elements[r][c] -= matrix.elements[r][c]
	return self


## function to multiply every element of the matrix by -1
func negate() -> Matrix:
	if !is_valid():
		push_error("Negation of matrices: The matrix is invalid")
		return self
	
	for r in range(rows):
		for c in range(columns):
			elements[r][c] *= -1
	
	return self


## computes the sum of all elements on the diagonal of square matrices
func trace() -> float:
	if !is_valid():
		push_error("trace(): Matrix is invalid")
		return -1
	
	if !is_square():
		push_error("trace(): Trace is defined only fro square matrices")
		return -1
	
	var sum : float = 0.0
	for i in range(rows):
		sum += elements[i][i]
	
	return sum


## Returns a new matrix with the specified row removed.
func delete_row(index: int) -> Matrix:
	if !is_valid():
		push_error("delete_row(): Matrix is not valid.")
		return self
	
	if index < 0 or index >= rows:
		push_error("delete_row(): Row index out of bounds.")
		return self
	
	var new_elements = []
	for r in range(rows):
		if r == index:
			continue
		new_elements.append(elements[r].duplicate(true))
	
	return Matrix.new(new_elements)


## Returns a new matrix with the specified column removed.
func delete_column(index: int) -> Matrix:
	if !is_valid():
		push_error("delete_column(): Matrix is not valid.")
		return self
	
	if index < 0 or index >= columns:
		push_error("delete_column(): Column index out of bounds.")
		return self
	
	var new_elements = []
	for r in range(rows):
		var new_row = []
		for c in range(columns):
			if c == index:
				continue
			new_row.append(elements[r][c])
		new_elements.append(new_row)
	
	return Matrix.new(new_elements)


## Returns the minor matrix by deleting the specified row and column.
func minor_matrix(row_index: int, col_index: int) -> Matrix:
	return delete_row(row_index).delete_column(col_index)


## Calculates the determinant of the matrix using recursive Laplace expansion.
func determinant() -> float:
	if !is_valid():
		push_error("determinant(): Matrix is not valid.")
		return 0.0
	
	if !is_square():
		push_error("determinant(): Matrix must be square.")
		return 0.0
	
	# Base case: 1x1 matrix
	if rows == 1:
		return elements[0][0]
	
	# Base case: 2x2 matrix
	if rows == 2:
		return elements[0][0] * elements[1][1] - elements[0][1] * elements[1][0]
	
	# Recursive case: expand along first row
	var det = 0.0
	for col in range(columns):
		var sign_value = (-1.0) ** col
		var sub = minor_matrix(0, col)
		det += sign_value * elements[0][col] * sub.determinant()
	
	return det


## Calculates and returns the inverse of this matrix (if invertible).
## Uses: A⁻¹ = (1/det(A)) × adj(A) = (1/det(A)) × cofactor(A).transposed()
func inverse() -> Matrix:
	if !is_valid():
		push_error("inverse(): Matrix is not valid.")
		return self

	if !is_square():
		push_error("inverse(): Matrix must be square.")
		return self

	var det = determinant()
	if abs(det) < core.EPSILON:
		push_error("inverse(): Matrix is not invertible (determinant is zero).")
		return self

	var cofactor_mat = cofactor()
	var adjugate = cofactor_mat.transposed()
	var inverse_mat = adjugate.multiply_by(1.0 / det)

	return inverse_mat


## Swaps two rows of the matrix.
func swap_rows(r1: int, r2: int) -> Matrix:
	if !is_valid():
		push_error("swap_rows(): Matrix is not valid.")
		return self
	
	if r1 < 0 or r1 >= rows or r2 < 0 or r2 >= rows:
		push_error("swap_rows(): Row indices out of bounds.")
		return self
	
	var temp = elements[r1]
	elements[r1] = elements[r2]
	elements[r2] = temp
	
	return self


## Multiplies all elements in a row by a non-zero factor.
func scale_row(r: int, factor: float) -> Matrix:
	if !is_valid():
		push_error("scale_row(): Matrix is not valid.")
		return self

	if r < 0 or r >= rows:
		push_error("scale_row(): Row index out of bounds.")
		return self
	
	if abs(factor) < core.EPSILON:
		push_error("scale_row(): Cannot scale by zero.")
		return self

	for c in range(columns):
		elements[r][c] *= factor

	return self


## Adds a scaled version of 'source' row to 'target' row.
## That is: target = target + source * factor
func add_rows(target: int, source: int, factor: float) -> Matrix:
	if !is_valid():
		push_error("add_rows(): Matrix is not valid.")
		return self
	
	if target < 0 or target >= rows or source < 0 or source >= rows:
		push_error("add_rows(): Row indices out of bounds.")
		return self
	
	for c in range(columns):
		elements[target][c] += elements[source][c] * factor
	
	return self


## Replaces all near-zero float values with 0.0 using core.EPSILON.
func clean_near_zero() -> Matrix:
	if !is_valid():
		push_error("clean_near_zero(): Matrix is not valid.")
		return self
	
	for r in range(rows):
		for c in range(columns):
			if abs(elements[r][c]) < core.EPSILON:
				elements[r][c] = 0.0
	
	return self


## Reduces the matrix to row echelon form using Gaussian elimination.
## This function modifies self and returns self for chaining.
func gauss_reduce() -> Matrix:
	if !is_valid():
		push_error("gauss_reduce(): Matrix is not valid.")
		return self
	
	var lead = 0  # column we're reducing
	for r in range(rows):
		if lead >= columns:
			break
		
		var i = r
		while abs(elements[i][lead]) < core.EPSILON: # abs(smth) will always be > 0 so if abs(...) is between 0 and EPSILON it's treated as zero
			i += 1
			if i == rows:
				i = r
				lead += 1
				if lead == columns:
					return self
		swap_rows(i, r)
		
		var pivot = elements[r][lead]
		if abs(pivot) > core.EPSILON:
			scale_row(r, 1.0 / pivot)
	
		for j in range(rows):
			if j != r:
				var factor = -elements[j][lead]
				add_rows(j, r, factor)
	
		lead += 1
	
	return self


## Solves the linear system Ax = B using Gaussian elimination.
## Returns the solution matrix X if solvable.
func solve(B: Matrix) -> Matrix:
	if !is_valid() or !B.is_valid():
		push_error("solve(): One of the matrices is invalid.")
		return self
	
	if rows != B.rows:
		push_error("solve(): Incompatible dimensions: A.rows must equal B.rows.")
		return self
	
	# Create augmented matrix [A | B]
	var augmented = Matrix.generate_matrix(rows, columns + B.columns, 0.0)
	for r in range(rows):
		for c in range(columns):
			augmented.elements[r][c] = elements[r][c]
		for bc in range(B.columns):
			augmented.elements[r][columns + bc] = B.elements[r][bc]
	
	# Perform Gaussian elimination on augmented matrix
	augmented.gauss_reduce().clean_near_zero()
	
	# Extract solution columns (everything after columns of A)
	var result = Matrix.generate_matrix(columns, B.columns, 0.0)
	for r in range(columns):
		for bc in range(B.columns):
			result.elements[r][bc] = augmented.elements[r][columns + bc]
	
	return result



## Applies a function to each element of the matrix and returns a new Matrix.
## The function must accept a float and return a float.
func map(func_ref: Callable) -> Matrix:
	if !is_valid():
		push_error("map(): Matrix is not valid.")
		return self
	
	if !func_ref.is_valid():
		push_error("map(): Function is not valid.")
		return self
	
	for r in range(rows):
		for c in range(columns):
			elements[r][c] = func_ref.call(elements[r][c])
	
	return self

#endregion


#region UTILS
## function to print the rows of an instance of Matrix
static func printm(matrix : Matrix) -> void:
	for i in range(matrix.rows):
		print(matrix.elements[i])


## helps debug errors with matrices
static func debug_shape(matrix: Matrix, label: String = "") -> void:
	var prefix = label + ": " if label != "" else ""
	print("%sMatrix %s | r*c: %s×%s | Valid: %s" % [
		prefix, matrix, matrix.rows, matrix.columns, matrix.is_valid()
	])
#endregion
