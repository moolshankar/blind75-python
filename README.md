# blind75-python
Blind 75 problem with solutions in single file


## Array and Maps

### 1. [Leetcode 217](https://leetcode.com/problems/contains-duplicate/description) : Contains Duplicate
<pre>
Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

Example 1:

Input: nums = [1,2,3,1]
Output: true

Explanation:
The element 1 occurs at the indices 0 and 3.

Example 2:

Input: nums = [1,2,3,4]
Output: false

Explanation:
All elements are distinct.

Example 3:

Input: nums = [1,1,1,3,3,4,3,2,4,2]
Output: true
</pre>

Solution:

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        _set = set()
        for num in nums:
            if num in _set:
                return True
            _set.add(num)
        return False
```

### 2. [Leetcode 242](https://leetcode.com/problems/valid-anagram/description/) : Valid Anagram
<pre>
Given two strings s and t, return true if t is an 
anagram of s, and false otherwise.

 

Example 1:

Input: s = "anagram", t = "nagaram"
Output: true

Example 2:

Input: s = "rat", t = "car"
Output: false

Constraints:

1 <= s.length, t.length <= 5 * 104
s and t consist of lowercase English letters.
</pre>

Solution:

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        _list_s = [0] * 26
        _list_t = [0] * 26
        if len(s) != len(t):
            return False
        for i in range(len(s)):
            _list_s[ord(s[i]) - ord('a')] += 1
            _list_t[ord(t[i]) - ord('a')] += 1
        if tuple(_list_s) == tuple(_list_t):
            return True
        return False
```

### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 4. [Leetcode 49](https://leetcode.com/problems/group-anagrams/description/) : Group Anagrams
<pre>
Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]

Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Explanation:

There is no string in strs that can be rearranged to form "bat".
The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.
The strings "ate", "eat", and "tea" are anagrams as they can be rearranged to form each other.
Example 2:

Input: strs = [""]

Output: [[""]]

Example 3:

Input: strs = ["a"]

Output: [["a"]]

 

Constraints:

1 <= strs.length <= 104
0 <= strs[i].length <= 100
strs[i] consists of lowercase English letters.
</pre>

Solution:

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        _dict = {}
        for _str in strs:
            _list = [0] * 26
            for ch in _str:
                i = ord(ch) - ord('a') 
                _list[i] += 1
            k = tuple(_list)
            if k not in _dict:
                _dict[k] = []
            _dict[k].append(_str)
        return list(_dict.values())
```


### 5. [Leetcode 347](https://leetcode.com/problems/top-k-frequent-elements/description/) : Top K Frequent Elements
<pre>
Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

 

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
Example 2:

Input: nums = [1], k = 1
Output: [1]
 

Constraints:

1 <= nums.length <= 105
-104 <= nums[i] <= 104
k is in the range [1, the number of unique elements in the array].
It is guaranteed that the answer is unique.
 

Follow up: Your algorithm's time complexity must be better than O(n log n), where n is the array's size. 
</pre>

Solution:

```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        _dict_count = {}
        _dict_group = {}
        countMax = 0
        for n in nums:
            _dict_count[n] = _dict_count.get(n, 0) + 1
        for _k,v in _dict_count.items():
            if v not in _dict_group:
                _dict_group[v] = []
            _dict_group[v].append(_k)
            if v > countMax:
                countMax = v
        res = []
        list_count = 0
        for i in range(countMax, -1, -1):
            if i in _dict_group:
                res.extend(_dict_group[i])
                list_count += len(_dict_group[i])
            if list_count == k:
                break
        return res
```


### 6. [Leetcode 271 : Premium](https://leetcode.com/problems/encode-and-decode-strings/description/) : Encode and Decode Strings
<pre>
Design an algorithm to encode a list of strings to a single string. The encoded string is then decoded back to the original list of strings.

Please implement encode and decode

Example 1:

Input: ["neet","code","love","you"]

Output:["neet","code","love","you"]
Example 2:

Input: ["we","say",":","yes"]

Output: ["we","say",":","yes"]
Constraints:

0 <= strs.length < 100
0 <= strs[i].length < 200
strs[i] contains only UTF-8 characters.
 
</pre>

Solution:

```python
class Solution:
    def encode(self, strs: List[str]) -> str:
        res = ""
        delimiter = "#"
        for word in strs:
            _len = len(word)
            res = res + str(_len) + delimiter + word
        return res

    def decode(self, s: str) -> List[str]:
        res = []
        i = 0
        while i < len(s):
            _lenS = ""
            while s[i] != "#":
                _lenS = _lenS + s[i]
                i += 1
            start = i+1
            end = start + int(_lenS)
            res.append(s[start:end])
            i = end
        return res
```


### 7. [Leetcode 238](https://leetcode.com/problems/product-of-array-except-self/description/) : Product of Array Except Self
<pre>
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.

 

Example 1:

Input: nums = [1,2,3,4]
Output: [24,12,8,6]
Example 2:

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
 

Constraints:

2 <= nums.length <= 105
-30 <= nums[i] <= 30
The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
 

Follow up: Can you solve the problem in O(1) extra space complexity? (The output array does not count as extra space for space complexity analysis.) 
</pre>

Solution:

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        _len = len(nums)
        _res = [1] * _len
        _mul_left = 1
        for i in range(_len):
            _mul_left *= nums[i]
            _res[i] = _mul_left
        _mul_right = 1
        for i in range(_len):
            if i != 0:
                _mul_right *= nums[_len-i]
            if i == _len-1:
                _res[_len-1-i] = _mul_right
            else:
                _res[_len-1-i] = _res[_len-2-i] * _mul_right
        return _res
```


### 8. [Leetcode 128](https://leetcode.com/problems/longest-consecutive-sequence/description/) : Longest Consecutive Sequence
<pre>
Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.

 

Example 1:

Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
Example 2:

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
 

Constraints:

0 <= nums.length <= 105
-109 <= nums[i] <= 109 
</pre>

Solution:

```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        _set = set(nums)
        _max = 0
        for num in _set:
            if num-1 in _set:
                continue
            _count = 0
            n = num
            while n in _set:
                _count += 1
                n += 1
            if _count > _max:
                _max = _count
        return _max
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum
<pre>
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.


Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]
Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]
 

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.
 

Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity? 
</pre>

Solution:

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        _dict = {}
        for i in range(len(nums)):
            num = nums[i]
            if num in _dict:
                return [_dict[num], i]
            _dict[target-num] = i
        return []
```
