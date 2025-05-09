extends PanelContainer

var open_network : NeuralNetwork

# Called when the node enters the scene tree for the first time.
func _ready():
	($MarginContainer/Control/Body/HSplitContainer as SplitContainer).split_offset = (get_viewport_rect() as Rect2).size.x/2
	($MarginContainer/Control/Body/HSplitContainer as SplitContainer).collapsed = false
	var XOR_Network : NeuralNetwork = NeuralNetwork.from_dict(core.load_dict("user://XOR_neural_network.json"))
	var input : Matrix = Matrix.new([[1.0],[2.0]])
	print("Input Matrix:")
	Matrix.printm(input)
	print("Output Matrix:")
	Matrix.printm(XOR_Network.forward(input))
	
	

@warning_ignore("unused_parameter")
# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass


func _on_back_button_pressed():
	get_tree().change_scene_to_file("res://Main/Scenes/main.tscn")


func _on_file_dialog_file_selected(path):
	($MarginContainer/Control/FileControls/FileEdit as TextEdit).text = path
	_on_file_edit_text_changed()


func _on_file_button_pressed():
	($MarginContainer/Control/FileControls/FileDialog as FileDialog).popup()


func _on_file_edit_text_changed():
	pass # Replace with function body.
