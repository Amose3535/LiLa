extends Control


# Called when the node enters the scene tree for the first time.
func _ready():
	pass


@warning_ignore("unused_parameter")
# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass


func _on_ml_button_pressed():
	get_tree().change_scene_to_file("res://Main/Scenes/ML_Demo.tscn")
