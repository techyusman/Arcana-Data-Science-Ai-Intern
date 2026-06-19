# Python has Dynamic Variable capability it will recoginze the data type automatically at run time.
name = "M Usman"
age = 21
student = True

# For Print we use these statements 

print((name))
print((age))

# to check the type of variable
print(type(student))

# Changing its type 
new_age = "21"
age = int(new_age)
print(type(age))


# Control flow allows programs to make decisions.

if age >= 18:
    print("Eligible for driving")
else:
    print("Not Eligible")

# we will have if condition, if else and if elif else. 




#For Loop Used when number of iterations is known.

# For Loop
for i in range(5):
    print(i)

# While Loop Used when number of iterations is unknown.

count = 1 
while (count <= 5):
    print(count)
    count +=1


#Functions help avoid repetition. create to perform specific task.
#function without parameter
def greet():
    print("Hello Usman")
    print("Welcome to Arcana")

greet()

#function with parameters
def greet(name):
    print("Hello"+name)

greet("M.Usman")

#function with return value
def add(a,b):
    return a+b

sum = add(4,5)
print(sum)

#Scope - local and global

#Global variable
global_var = 100
def test():
    print(global_var)  # Accessing Global Variable

test()

#local variable
def test():
    y = 20 #local variable
    print(y)

#print(y) #will show error because of scope
test()


#Lists, Dictionaries & Sets

#Lists Store multiple values.
students = ["Ali","Ahmed","Usman"]
print(students[0]) #access through index

#Dictionaries Store key-value pairs. 
student = {"name":"Usman","age":21}
print(student["name"]) #access through key

#Sets Store unique values.
numbers = {1,2,3,4,4,5}
print(numbers) #will print only unique values

# File I/O Basics
# Reading Files
file = open("data.txt","r")
content = file.read()
print(content)
file.close()

#Writing Files
file = open("data.txt","w")
file.write("Hello World")
file.close()

#Better approach 
with open("data.txt","r") as file:
    content = file.read()
print(content)
#it Automatically closes the file.

