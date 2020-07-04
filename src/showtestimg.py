from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import json

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# pick a sample to plot
sample = 164
image = X_train[sample]
image_array = image.reshape((1,784))
image_array_decimal = image_array/255
lists = image_array_decimal.tolist()
json_str = json.dumps(lists)
print(json_str)
# plot the sample
fig = plt.figure
plt.imshow(image, cmap='gray')
plt.show()