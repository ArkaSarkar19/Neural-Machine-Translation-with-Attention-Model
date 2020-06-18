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
   * After training for 20 epochs. </br>
   
            source: 3 May 1979
            output: 1979-05-03
            source: 5 April 09
            output: 2009-04-05
            source: 21th of August 2016
            output: 2016-08-01
            source: Tue 10 Jul 2007
            output: 2007-07-10
            source: Saturday May 9 2018
            output: 2018-05-09
            source: March 3 2001
            output: 2001-03-03
            source: March 3rd 2001
            output: 2001-03-01
            source: 1 March 2001
            output: 2001-03-01



### Note
The code might take a couple of minutes to run as it trains 20 epochs. </br>
