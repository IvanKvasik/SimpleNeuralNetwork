# SimpleNeuralNetwork
A simple implementation of basic neural networks without dependencies in Python
## Usage
### Network class
The constructor accepts following parameters:
1. ```layers``` — a list of integers describing every layer of your network
2. ```wrapper``` — an instance of a ```Wrapper``` child class


You may load an existing model from a JSON-file using the ```load_model``` class method that accepts ```path```, the path to the model file, and ```wrapper```
#### Methods:
1. ```train(self, dataset, learning_rate, epochs, validation, patience, batch_size)```
    1. ```dataset``` — a list of ```(input_data, output_data)``` tuples
    2. ```learning_rate``` — a float number, may be reduced if ```validation``` and ```patience``` are set
    3. ```epochs``` — an integer
    4. ```validation``` *(optional)* — validation set in the format of dataset, used for early stopping and reducing learning rate if set
    5. ```patience``` *(optional)* — the number of epochs with no accuracy improvments needed for stopping the training early
    6. ```batch_size``` *(optional)* — an integer
2. ```get_answer(self, input_data)```
    1. ```input_data``` — a list of float numbers
 

    Returns a list of float numbers — the calculated answer
3. ```save_model(self, path)```
    1. ```path``` — the path to save the file

### Wrapper class
An abstract class used for easily defining external activation functions requiring following methods:
1. ```function(self, x: float) -> float``` — the actual function
2. ```derivative(self, x: float) -> float``` — the function's derivative

Two basic ```Wrapper``` classes ```Sigmoid``` and ```Linear``` are available from the box.
