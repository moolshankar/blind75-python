# blind75-python
Blind 75 problems with solution in single file


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


## Two Pointers


### 9. [Leetcode 125](https://leetcode.com/problems/valid-palindrome/description/) : Valid Palindrome
<pre>
A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise.

 

Example 1:

Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.
Example 2:

Input: s = "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.
Example 3:

Input: s = " "
Output: true
Explanation: s is an empty string "" after removing non-alphanumeric characters.
Since an empty string reads the same forward and backward, it is a palindrome.
 

Constraints:

1 <= s.length <= 2 * 105
s consists only of printable ASCII characters. 
</pre>

Solution:

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = s.lower()
        i = 0
        j = len(s) - 1
        while i < j:
            if not s[i].isalnum():
                i += 1
                continue
            if not s[j].isalnum():
                j -= 1
                continue
            if s[i] != s[j]:
                return False
            i += 1
            j -= 1
        return True
```


### 10. [Leetcode 15](https://leetcode.com/problems/3sum/description/) : 3Sum
<pre>
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.

 

Example 1:

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
Explanation: 
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].
Notice that the order of the output and the order of the triplets does not matter.
Example 2:

Input: nums = [0,1,1]
Output: []
Explanation: The only possible triplet does not sum up to 0.
Example 3:

Input: nums = [0,0,0]
Output: [[0,0,0]]
Explanation: The only possible triplet sums up to 0.
 

Constraints:

3 <= nums.length <= 3000
-105 <= nums[i] <= 105
</pre>

Solution:

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums = sorted(nums)
        _len = len(nums)
        if _len < 3:
            return []
        res = []
        for i in range(_len):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            j = i + 1
            k = _len - 1
            while j < k :
                while nums[i] + nums[j] + nums[k] > 0 and k > j:
                    k -= 1
                while nums[i] + nums[j] + nums[k] < 0 and k > j:
                    j += 1
                if nums[i] + nums[j] + nums[k] == 0 and i < j < k:
                    res.append([nums[i], nums[j], nums[k]])
                    j += 1
                    while nums[j] == nums[j-1] and j < k:
                        j += 1
        return res
```


### 11. [Leetcode 11](https://leetcode.com/problems/container-with-most-water/description/) : Container With Most Water
<pre>
You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.

 

Example 1:


Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
Example 2:

Input: height = [1,1]
Output: 1
 

Constraints:

n == height.length
2 <= n <= 105
0 <= height[i] <= 104
</pre>

Solution:

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        _max = 0
        l = 0
        r = len(height) - 1
        while l < r:
            if height[l] <= height[r]:
                _max = max(_max, (height[l] * (r-l)))
                l += 1
            else:
                _max = max(_max, (height[r] * (r-l)))
                r -= 1
        return _max
                
```

## Sliding Window

### 12. [Leetcode 121](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/description/) : Best Time to Buy and Sell Stock
<pre>
You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

 

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
Example 2:

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: In this case, no transactions are done and the max profit = 0.
 

Constraints:

1 <= prices.length <= 105
0 <= prices[i] <= 104 
</pre>

Solution:

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        left, right = 0, 0
        res = 0
        while right < len(prices):
            res = max(res, prices[right] - prices[left])
            if prices[right] < prices[left]:
                left = right
            right += 1
        return res
```


### 13. [Leetcode 3](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/) : Longest Substring Without Repeating Characters
<pre>
Given a string s, find the length of the longest 
substring
 without repeating characters.

 

Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
 

Constraints:

0 <= s.length <= 5 * 104
s consists of English letters, digits, symbols and spaces. 
</pre>

Solution:

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        memo = set()
        left, right, res = 0, 0, 0
        while right < len(s):
            ch = s[right]
            while ch in memo:
                memo.remove(s[left])
                left += 1
            memo.add(ch)
            right += 1
            res = max(res, right-left)
        return res
```


### 14. [Leetcode 424](https://leetcode.com/problems/longest-repeating-character-replacement/description/) : Longest Repeating Character Replacement
<pre>
You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.

 

Example 1:

Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.
Example 2:

Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".
The substring "BBBB" has the longest repeating letters, which is 4.
There may exists other ways to achieve this answer too.
 

Constraints:

1 <= s.length <= 105
s consists of only uppercase English letters.
0 <= k <= s.length
</pre>

Solution:

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        l = 0
        res = 0
        _dict = {}
        for r in range(len(s)):
            _dict[s[r]] = 1 + _dict.get(s[r], 0) 
            while (r - l + 1) - max(_dict.values()) > k:
                _dict[s[l]] -= 1
                l +=1
            res = max(res, (r - l + 1))
        return res

```


### 15. [Leetcode 76](https://leetcode.com/problems/minimum-window-substring/description/) : Minimum Window Substring
<pre>
Given two strings s and t of lengths m and n respectively, return the minimum window 
substring
 of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".

The testcases will be generated such that the answer is unique.

 

Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
Example 2:

Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.
Example 3:

Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.
 

Constraints:

m == s.length
n == t.length
1 <= m, n <= 105
s and t consist of uppercase and lowercase English letters.
 

Follow up: Could you find an algorithm that runs in O(m + n) time? 
</pre>

Solution:

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        t_map = {}
        s_map = {}
        for ch in t:
            t_map[ch] = 1 + t_map.get(ch, 0)
            s_map[ch] = 0
        need, have = len(t_map), 0
        left = 0
        res = ""
        for right in range(len(s)):
            ch = s[right]
            if ch not in s_map:
                continue
            s_map[ch] = 1 + s_map[ch]
            if s_map[ch] == t_map[ch]:
                have += 1
            while have == need:
                while s[left] not in s_map:
                    left += 1
                if res == "" or (right - left + 1) < len(res):
                    res = s[left:right+1]
                s_map[s[left]] -= 1
                if s_map[s[left]] < t_map[s[left]]:
                    have -= 1
                left += 1
        return res
```

## Stack

### 16. [Leetcode 20](https://leetcode.com/problems/valid-parentheses/description/) : Valid Parentheses
<pre>
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.
 

Example 1:
Input: s = "()"
Output: true

Example 2:
Input: s = "()[]{}"
Output: true

Example 3:
Input: s = "(]"
Output: false

Example 4:
Input: s = "([])"
Output: true


Constraints:

1 <= s.length <= 104
s consists of parentheses only '()[]{}'.
</pre>

Solution:

```python
class Solution:
    def isValid(self, s: str) -> bool:
        _list = []
        for ch in s:
            if ch == ')' and (len(_list) == 0 or _list.pop() != '('):
                return False
            elif ch == '}' and (len(_list) == 0 or _list.pop() != '{'):
                return False
            elif ch == ']' and (len(_list) == 0 or _list.pop() != '['):
                return False
            elif ch == '(' or ch == '{' or ch == '[':
                _list.append(ch)
        return len(_list) == 0
```

## Binary Search


### 17. [Leetcode 153](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/) : Find Minimum in Rotated Sorted Array
<pre>
Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

[4,5,6,7,0,1,2] if it was rotated 4 times.
[0,1,2,4,5,6,7] if it was rotated 7 times.
Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

 

Example 1:

Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.
Example 2:

Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.
Example 3:

Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 
 

Constraints:

n == nums.length
1 <= n <= 5000
-5000 <= nums[i] <= 5000
All the integers of nums are unique.
nums is sorted and rotated between 1 and n times. 
</pre>

Solution:

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        l= 0
        r = len(nums)-1
        res = nums[l]
        while l <= r:
            if nums[l] > nums[r]:
                res = min(res, nums[r])
                l += 1
            else:
                res = min(res, nums[l])
                r -= 1
        return res
```


### 18. [Leetcode 33](https://leetcode.com/problems/search-in-rotated-sorted-array/description/) : Search in Rotated Sorted Array
<pre>
There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
Example 3:

Input: nums = [1], target = 0
Output: -1
 

Constraints:

1 <= nums.length <= 5000
-104 <= nums[i] <= 104
All values of nums are unique.
nums is an ascending array that is possibly rotated.
-104 <= target <= 104
</pre>

Solution:

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        l = 0
        r = len(nums) - 1
        while l <= r:
            mid = l + ((r-l) // 2)
            if target == nums[mid]:
                return mid
            if  nums[l] <= nums[mid]:
                if target < nums[l] or target > nums[mid]:
                    l = mid + 1
                else:
                    r = mid - 1
            else:
                if target > nums[r] or target < nums[mid]:
                    r = mid - 1
                else:
                    l = mid + 1
        return -1
```

## Linked List

### 19. [Leetcode 206](https://leetcode.com/problems/reverse-linked-list/description/) : Reverse Linked List
<pre>
Given the head of a singly linked list, reverse the list, and return the reversed list.

 

Example 1:


Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]
Example 2:


Input: head = [1,2]
Output: [2,1]
Example 3:

Input: head = []
Output: []
 

Constraints:

The number of nodes in the list is the range [0, 5000].
-5000 <= Node.val <= 5000
 

Follow up: A linked list can be reversed either iteratively or recursively. Could you implement both? 
</pre>

Solution:

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # iterative method
        # prev, curr = None, head
        # while curr:
        #     nxt = curr.next
        #     curr.next = prev
        #     prev = curr
        #     curr = nxt
        # return prev

        # recursion
        if head == None or head.next == None:
            return head
        newHead = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return newHead
```


### 20. [Leetcode 21](https://leetcode.com/problems/merge-two-sorted-lists/description/) : Merge Two Sorted Lists
<pre>
You are given the heads of two sorted linked lists list1 and list2.

Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list.

 

Example 1:


Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]
Example 2:

Input: list1 = [], list2 = []
Output: []
Example 3:

Input: list1 = [], list2 = [0]
Output: [0]
 

Constraints:

The number of nodes in both lists is in the range [0, 50].
-100 <= Node.val <= 100
Both list1 and list2 are sorted in non-decreasing order.
</pre>

Solution:

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        list3 = ListNode()
        tail = list3
        while list1 and list2:
            if list1.val <= list2.val:
                tail.next = list1
                list1 = list1.next
            else:
                tail.next = list2
                list2 = list2.next
            tail = tail.next
        if list1:
            tail.next = list1
        if list2:
            tail.next = list2
        return list3.next
```


### 21. [Leetcode 143](https://leetcode.com/problems/reorder-list/description/) : Reorder List
<pre>
You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln
Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
You may not modify the values in the list's nodes. Only nodes themselves may be changed.

 

Example 1:


Input: head = [1,2,3,4]
Output: [1,4,2,3]
Example 2:


Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]
 

Constraints:

The number of nodes in the list is in the range [1, 5 * 104].
1 <= Node.val <= 1000
</pre>

Solution:

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        slow, fast, start = head, head, head
        while fast:
            if fast.next is None or fast.next.next is None:
                slow = slow.next
                break
            slow = slow.next
            fast = fast.next.next
        slow = self.reverseList(slow)
        middle = slow
        dummy = ListNode()
        tail = dummy
        while start != middle:
            tail.next = start
            start = start.next
            tail = tail.next
            tail.next = slow
            if slow is None:
                break
            slow = slow.next
            tail = tail.next
        return dummy.next
```


### 22. [Leetcode 19](https://leetcode.com/problems/remove-nth-node-from-end-of-list/description/) : Remove Nth Node From End of List
<pre>
Given the head of a linked list, remove the nth node from the end of the list and return its head.

Example 1:

Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
Example 2:

Input: head = [1], n = 1
Output: []
Example 3:

Input: head = [1,2], n = 1
Output: [1]
 

Constraints:

The number of nodes in the list is sz.
1 <= sz <= 30
0 <= Node.val <= 100
1 <= n <= sz
 

Follow up: Could you do this in one pass? 
</pre>

Solution:

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        slow, fast = dummy, head
        while n > 0 and fast:
            fast = fast.next
            n -= 1
        while fast:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return dummy.next
```


### 23. [Leetcode 141](https://leetcode.com/problems/linked-list-cycle/description/) : Linked List Cycle
<pre>
Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.

 

Example 1:


Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).
Example 2:


Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.
Example 3:


Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.
 

Constraints:

The number of the nodes in the list is in the range [0, 104].
-105 <= Node.val <= 105
pos is -1 or a valid index in the linked-list.
 

Follow up: Can you solve it using O(1) (i.e. constant) memory?
</pre>

Solution:

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if head == None:
            return False
        slow, fast = head, head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False
```

### 24. [Leetcode 23](https://leetcode.com/problems/merge-k-sorted-lists/description/) : Merge k Sorted Lists

<pre>
You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

 

Example 1:

Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6
Example 2:

Input: lists = []
Output: []
Example 3:

Input: lists = [[]]
Output: []
 

Constraints:

k == lists.length
0 <= k <= 104
0 <= lists[i].length <= 500
-104 <= lists[i][j] <= 104
lists[i] is sorted in ascending order.
The sum of lists[i].length will not exceed 104
</pre>

Solution:

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        if len(lists) == 0:
            return None
        def mergeTwoLists(node1, node2):
            merged = ListNode()
            tail = merged
            while node1 and node2:
                if node1.val < node2.val:
                    tail.next = node1
                    node1 = node1.next
                else:
                    tail.next = node2
                    node2 = node2.next
                tail = tail.next
            if node1:
                tail.next = node1
            if node2:
                tail.next = node2
            return merged.next

        while len(lists) > 1:
            mergedList = []
            for i in range(0, len(lists), 2):
                list1 = lists[i]
                list2 = lists[i+1] if (i+1) < len(lists) else None
                mergedList.append(mergeTwoLists(list1, list2))
            lists = mergedList
        return lists[0]
```

## Trees

### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```

### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```

### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```


### 3. [Leetcode 1](https://leetcode.com/problems/two-sum/description/) : Two Sum

<pre>

</pre>

Solution:

```python


```

