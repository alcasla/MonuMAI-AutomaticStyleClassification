"""
# calculate predictions
predict_probs = model.predict(test_data, batch_size=1)
predict_class = model.predict_classes(test_data, batch_size=1)

#load model from json and create
json = open('model.json', 'r')
model_json = json.read()
json.close()
loaded_model = model_from_json(model_json)
# load weights into loaded model
loaded_model.load_weights('model.h5')
print('Model loaded')

# Evaluate model
loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
loss, accuracy = loaded_model.evaluate(test_data, hotone_test_labels, verbose=1)
print('Loaded model. Test loss: ', loss, '\tTest accuracy: ', accuracy)
"""