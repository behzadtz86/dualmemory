 ## Dual-Channel Architectures in Incremental Learning (Tensorflow v2.2)
 - #### SOM Training
 - #### DNN for training the distances (from SOM)
 - #### Evaluating the model performance 
 
#### Running the Script:
`python3 main.py <options>`

example 1: `python3 main.py --x 10 --y 10 --d1 0,1,2,3,4 --d2 5,6,7,8,9,0 --plot_som --limit 200`

example 2: `--d1 0,1,2,3,4,5,6,7,8 --d2 9 --batch 100 --radius 5,1.5 --lr 0.5,0.01 --image_path ~/plots`

<table>
<thead>
<tr>
  <th>Option</th>
  <th>Type</th>
  <th>Explanation</th>
</tr>
<thead>
<tbody>
<tr>
  <td>--batch</td>
  <td>int (default=1)</td>
  <td>The size of batches</td>
</tr>
 

<tr>
  <td>--x</td>
  <td>int (default=10)</td>
  <td>SOM Width</td>
</tr>

<tr>
  <td>--y</td>
  <td>int (default=10)</td>
  <td>SOM Height</td>
</tr>

<tr>
  <td>--dnn_epoch</td>
  <td>str (default='1,1,1,1,1')</td>
 <td>Number of DNN epochs in each sub-task (separated by ',').</td>
</tr>

<tr>
  <td>--d1</td>
  <td>str (default=None)</td>
  <td>The list of classes for training in sub-task 1 (separated by ',').</td>
</tr>

<tr>
  <td>--d2</td>
  <td>str (default=None)</td>
  <td>The list of classes for training in sub-task 2 (separated by ',').</td>
</tr>

<tr>
  <td>--d3</td>
  <td>str (default=None)</td>
  <td>The list of classes for training in sub-task 3 (separated by ',').</td>
</tr>

<tr>
  <td>--d4</td>
  <td>str (default=None)</td>
  <td>The list of classes for training in sub-task 4 (separated by ',').</td>
</tr>

<tr>
  <td>--d5</td>
  <td>str (default=None)</td>
  <td>The list of classes for training in sub-task 5 (separated by ',').</td>
</tr>

<tr>
  <td>--limit</td>
  <td>int (default=50)</td>
  <td>Limits the number of samples per class.</td>
</tr>

<tr>
  <td>--radius</td>
  <td>str (default='5.0,0.5,0.5,0.5,0.5')</td>
  <td>The initial radius of SOM training in each sub-task (separated by ',').</td>
</tr>

<tr>
  <td>lr</td>
  <td>str (default='1,0.01,0.01,0.01,0.01')</td>
  <td>The initial learning-rate of SOM training in each sub-task (separated by ',').</td>
</tr>

<tr>
  <td>--image_path</td>
  <td>str (default=None)</td>
  <td>The folder path For saving the PNG/PDF plots</td>
</tr>

<tr>
  <td>--plot_som</td>
  <td>boolean (default=false)</td>
  <td>Plot The Result of SOM training</td>
</tr>
</tbody>
</table>     
