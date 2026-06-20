userinput = 0
data=[]


def showmenu():
    print("===== TO-DO LIST =====")
    print("1. Add Task")
    print("2. Mark as Done")
    print("3. View Task")
    print("4. Exit")

while userinput != 4:
    showmenu()
    
    userinput = int(input("Enter your choice: "))
    
    if userinput ==1:
        task = input("What you want to add in your to-do list?")
        data.append(task)
        print("added task ",task," successfully")

    elif userinput == 2:
        task = input("Enter task to mark as done :")

        for task in data :
            data.remove(task)
            print("task removed ",task, " successfully")
            break
        else :
            print("Task not found")


    elif userinput == 3:
        print("Your current To Do List:")
        for task in data :
            print(task)

    elif userinput==4:
        print("exit")

    else :
        print("Please Enter valid number from 1 to 4 only")
        