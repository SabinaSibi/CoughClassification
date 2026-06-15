#testing data

test = model.fit(test_data, test_labels, batch_size=16, epochs=20)

test_loss, test_accuracy = model.evaluate (test_data, test_labels)


# plot loss 
plt.plot(test.history['loss'], color = 'red')
plt.xlabel('epochs')
plt.ylabel('percentage/100')
plt.legend(['loss'])
plt.show()

#plot accuracy
plt.plot(test.history['accuracy'], color='green')
plt.xlabel('epochs')
plt.ylabel('percentage/100')
plt.legend(['accuracy'])
plt.show()
