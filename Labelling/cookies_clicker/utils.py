


'''
    Possible moves for the vehicle. 
    
        [steer, throttle]
    Possibele values : 
        [-1,1][0 | 0.4,1] 

'''

moves = [
    [0, -1], # move n. 0
    [0, -0.7], # move n. 1
    [0, -0.5], # move n. 2
    [0, -0.3], # move n. 3
    [0, 0], # move n. 4
    [0, 0.3], # move n. 5
    [0, 0.5], # move n. 6
    [0, 0.7], # move n. 7
    [0, 1], # move n. 8
    [0.3, -1], # move n. 9
    [0.3, -0.7], # move n. 10
    [0.3, -0.5], # move n. 11
    [0.3, -0.3], # move n. 12
    [0.3, 0], # move n. 13
    [0.3, 0.3], # move n. 14
    [0.3, 0.5], # move n. 15
    [0.3, 0.7], # move n. 16
    [0.3, 1], # move n. 17
    [0.5, -1], # move n. 18
    [0.5, -0.7], # move n. 19
    [0.5, -0.5], # move n. 20
    [0.5, -0.3], # move n. 21
    [0.5, 0], # move n. 22
    [0.5, 0.3], # move n. 23
    [0.5, 0.5], # move n. 24
    [0.5, 0.7], # move n. 25
    [0.5, 1], # move n. 26
    [0.7, -1], # move n. 27
    [0.7, -0.7], # move n. 28
    [0.7, -0.5], # move n. 29
    [0.7, -0.3], # move n. 30
    [0.7, 0], # move n. 31
    [0.7, 0.3], # move n. 32
    [0.7, 0.5], # move n. 33
    [0.7, 0.7], # move n. 34
    [0.7, 1], # move n. 35
    [1, -1], # move n. 36
    [1, -0.7], # move n. 37
    [1, -0.5], # move n. 38
    [1, -0.3], # move n. 39
    [1, 0], # move n. 40
    [1, 0.3], # move n. 41
    [1, 0.5], # move n. 42
    [1, 0.7], # move n. 43
    [1, 1], # move n. 44
]

'''
    Generates a list of possible moves given two lists of possible values
'''
def move_generator(
                        possible_accellerations = [0, 0.3, 0.5, 0.7, 1],
                        possible_steering= [-1,-0.7, -0.5, -0.3, 0, 0.3, 0.5, 0.7, 1]
                    ):
    print("[")
    index = 0
    for i in possible_accellerations:
        for x in possible_steering:
            print(f"    [{i}, {x}], # move n. {index}")
            index+=1
    print("]")




if __name__ == '__main__':
    move_generator()
    
