### Chapter 1: Important Programs in Python

#### 1. Introduction to Python Programming
- **Basic Syntax, Variables, and Data Types**
  - **Syntax**: Python is known for its readability and simple syntax. It uses indentation to define code blocks.
  - **Variables**: Used to store information to be referenced and manipulated in a program.
    ```python
    x = 10
    name = "Ayush"
    is_student = True
    ```
  - **Data Types**: Common data types in Python include integers (`int`), floating-point numbers (`float`), strings (`str`), and booleans (`bool`).
    ```python
    integer_var = 42
    float_var = 3.14
    string_var = "Hello, World!"
    boolean_var = False
    ```

- **Control Structures: if-else, loops**
  - **If-Else**: Used for conditional execution of code blocks.
    ```python
    age = 18
    if age >= 18:
        print("You are an adult.")
    else:
        print("You are a minor.")
    ```
  - **Loops**: Used for repeated execution of a block of code.
    - **For Loop**: Iterates over a sequence.
      ```python
      for i in range(5):
          print(i)
      ```
    - **While Loop**: Repeats as long as a condition is true.
      ```python
      count = 0
      while count < 5:
          print(count)
          count += 1
      ```

#### 2. Functions and Modules
- **Defining and Calling Functions**
  - Functions encapsulate a block of code to perform a specific task.
    ```python
    def greet(name):
        return f"Hello, {name}!"
    
    print(greet("Ayush"))
    ```
- **Importing and Using Modules**
  - Modules are files containing Python code, which can be imported into another script.
    ```python
    import math
    
    print(math.sqrt(16))  # Output: 4.0
    ```

#### 3. Data Structures
- **Lists**: Ordered, mutable collections.
  ```python
  fruits = ["apple", "banana", "cherry"]
  fruits.append("date")
  print(fruits)  # Output: ['apple', 'banana', 'cherry', 'date']
  ```
- **Tuples**: Ordered, immutable collections.
  ```python
  coordinates = (10, 20)
  print(coordinates[0])  # Output: 10
  ```
- **Dictionaries**: Key-value pairs, unordered.
  ```python
  student = {"name": "Ayush", "age": 21, "grade": "A"}
  print(student["name"])  # Output: Ayush
  ```
- **Sets**: Unordered collections of unique elements.
  ```python
  unique_numbers = {1, 2, 3, 2, 1}
  print(unique_numbers)  # Output: {1, 2, 3}
  ```
- **Comprehensions**: Create new lists, dictionaries, or sets in a concise way.
  ```python
  squares = [x**2 for x in range(10)]
  print(squares)  # Output: [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
  ```

#### 4. File Handling
- **Reading and Writing Files**
  ```python
  with open("example.txt", "w") as file:
      file.write("Hello, World!")
  
  with open("example.txt", "r") as file:
      content = file.read()
      print(content)  # Output: Hello, World!
  ```
- **Working with CSV and JSON Files**
  - **CSV Files**
    ```python
    import csv

    # Writing to a CSV file
    with open("data.csv", "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Age", "Grade"])
        writer.writerow(["Ayush", 21, "A"])

    # Reading from a CSV file
    with open("data.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            print(row)
    ```
  - **JSON Files**
    ```python
    import json

    data = {
        "name": "Ayush",
        "age": 21,
        "grade": "A"
    }

    # Writing to a JSON file
    with open("data.json", "w") as file:
        json.dump(data, file)

    # Reading from a JSON file
    with open("data.json", "r") as file:
        data = json.load(file)
        print(data)
    ```

#### 5. Error Handling
- **Exceptions and Error Handling**
  ```python
  try:
      result = 10 / 0
  except ZeroDivisionError:
      print("You cannot divide by zero.")
  finally:
      print("This block always executes.")
  ```
- **Debugging Techniques**
  - Use print statements to trace variables.
  - Use debugging tools like `pdb` or IDE integrated debuggers.

#### 6. Important Algorithms and Programs
- **Fibonacci Series**
  ```python
  def fibonacci(n):
      sequence = [0, 1]
      while len(sequence) < n:
          sequence.append(sequence[-1] + sequence[-2])
      return sequence
  
  print(fibonacci(10))  # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
  ```
- **Sorting Algorithms**
  - **Quick Sort**
    ```python
    def quicksort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quicksort(left) + middle + quicksort(right)

    print(quicksort([3, 6, 8, 10, 1, 2, 1]))  # Output: [1, 1, 2, 3, 6, 8, 10]
    ```
  - **Merge Sort**
    ```python
    def mergesort(arr):
        if len(arr) <= 1:
            return arr
        mid = len(arr) // 2
        left = mergesort(arr[:mid])
        right = mergesort(arr[mid:])

        return merge(left, right)

    def merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    print(mergesort([3, 6, 8, 10, 1, 2, 1]))  # Output: [1, 1, 2, 3, 6, 8, 10]
    ```
- **Searching Algorithms**
  - **Binary Search**
    ```python
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1

    print(binary_search([1, 2, 3, 4, 5, 6, 7], 4))  # Output: 3
    ```
- **Recursion**
  - Factorial of a number
    ```python
    def factorial(n):
        if n == 0:
            return 1
        else:
            return n * factorial(n - 1)

    print(factorial(5))  # Output: 120
    ```
