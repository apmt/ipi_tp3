def Yluminance ((b, g, r)):
	kr = 299
	kg = 587
	kb = 114
	y = (kr*r)+(kg*g)+(kb*b)
	y /= 1000
	return y