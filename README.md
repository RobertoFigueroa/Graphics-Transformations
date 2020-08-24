# Graphics-Transformations
Transformations

## How to use it

All you have to do is to uncomment some of the following lines in order to render different perspectives:

```python
#low-angle-shot
# r.lookAt(posModel, V3(0,-8,0))

#medium shot
# r.lookAt(posModel, V3(0,0,0))
# posModel = V3(0,0,-30)


# highangle
r.lookAt(posModel, V3(0,15,0))
posModel = V3(0,5,-20)

#dutch angle
# r.camRotation = V3(0,0,15)
# r.createViewMatrix()
# r.createProjectionMatrix()
# posModel = V3(0,0,-20)
```

and you will get this results:

### Low angle shot
![alt text](https://github.com/RobertoFigueroa/Graphics-Transformations/blob/master/lowangle.bmp?raw=true)

### Medium  shot
![alt text](https://github.com/RobertoFigueroa/Graphics-Transformations/blob/master/mediumshot.bmp?raw=true)

### High angle shot
![alt text](https://github.com/RobertoFigueroa/Graphics-Transformations/blob/master/highangle.bmp?raw=true)

### Dutch angle shot
![alt text](https://github.com/RobertoFigueroa/Graphics-Transformations/blob/master/dutchangle.bmp?raw=true)

