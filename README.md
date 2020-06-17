# Neural-Machine-Translation-with-Attention-Model
The Neural Machine Translation (NMT) model translates human-readable dates ("25th of Feb, 1999") into machine-readable dates ("1999-02-25") </br>
The code is in Python 3. </br>



### Model Overview and Implementation Details
**model.py** has 2 main functions : one_step_attention() and model(). </br>
* one_step_attention() computes : </br> 
    * [α<t,1>,α<t,2>,...,α<t,Tx>]the attention weights </br>
    * context^⟨t⟩: the context vector </br >

* model() function </br>
   * Model first runs the input through a Bi-LSTM to get [a<1>,a<2>,...,a<Tx>][a<1>,a<2>,...,a<Tx>].
   * Then, model calls one_step_attention() Ty times using a for loop. At each iteration of this loop:
       * It gives the computed context vector context<t>context<t> to the post-attention LSTM.
       * It runs the output of the post-attention LSTM through a dense layer with softmax activation.
       * The softmax generates a prediction ŷ 
  
  
![ntm](readme_images/attn_model.png)

### How to Run and Testing
**model.py** has the main code and just run it on the command line. </br>
In model.py I have added some examples as a test set. You can modify that to test your own examples. </br>

* Sample Output 

      source: 3 May 1979
      output: 1977-07-17
      source: 5 April 09
      output: 2988-07-28
      source: 21th of August 2016
      output: 2000-01-13
      source: Tue 10 Jul 2007
      output: 2000-07-22
      source: Saturday May 9 2018
      output: 2018-02-28
      source: March 3 2001
      output: 2000-03-13
      source: March 3rd 2001
      output: 2000-01-13
      source: 1 March 2001
      output: 2000-01-11


### Note
The code might take a couple of minutes to run as it trains 20 epochs. </br>
