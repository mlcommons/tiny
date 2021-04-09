### Pre-Trained Audio Wakeword Models
This folder contains preliminary versions of the reference model for audio wakewords.
* `aww_ref_model/` is a directory containing a pre-trained floating-point model in TensorFlow's SavedModel format.  This model achieves roughly 95-96% accuracy on the validation set (speech commands v2) and 92% accuracy on the test set.
* `aww_ref_model.tflite` is the same model, quantized and converted to TFLite format.  It accepts 10x49 8b MFCC spectrogram features as input and also achieves about 92% accuracy on the test set.