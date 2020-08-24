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

### Medium  shot

### High angle shot

### Dutch angle shot

