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

### 25. [Leetcode 226](https://leetcode.com/problems/invert-binary-tree/description/) : Invert Binary Tree

<pre>
Given the root of a binary tree, invert the tree, and return its root.

 

Example 1:


Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]
Example 2:


Input: root = [2,1,3]
Output: [2,3,1]
Example 3:

Input: root = []
Output: []
 

Constraints:

The number of nodes in the tree is in the range [0, 100].
-100 <= Node.val <= 100
</pre>

Solution:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root == None:
            return root
        if root.left:
            root.left = self.invertTree(root.left)
        if root.right:
            root.right = self.invertTree(root.right)
        root.left, root.right = root.right, root.left
        return root
```


### 26. [Leetcode 104](https://leetcode.com/problems/maximum-depth-of-binary-tree/description/) : Maximum Depth of Binary Tree

<pre>
Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

 

Example 1:


Input: root = [3,9,20,null,null,15,7]
Output: 3
Example 2:

Input: root = [1,null,2]
Output: 2
 

Constraints:

The number of nodes in the tree is in the range [0, 104].
-100 <= Node.val <= 100
</pre>

Solution:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return 0
        return (1 + max(self.maxDepth(root.left), self.maxDepth(root.right)))
```


### 27. [Leetcode 100](https://leetcode.com/problems/same-tree/description/) : Same Tree

<pre>
Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

 

Example 1:


Input: p = [1,2,3], q = [1,2,3]
Output: true
Example 2:


Input: p = [1,2], q = [1,null,2]
Output: false
Example 3:


Input: p = [1,2,1], q = [1,1,2]
Output: false
 

Constraints:

The number of nodes in both trees is in the range [0, 100].
-104 <= Node.val <= 104
</pre>

Solution:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p is None and q is None:
            return True
        if p is None or q is None:
            return False
        if p.val != q.val:
            return False
        return (self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right))
```


### 28. [Leetcode 572](https://leetcode.com/problems/subtree-of-another-tree/description/) : Subtree of Another Tree

<pre>
Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise.

A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.

 

Example 1:


Input: root = [3,4,5,1,2], subRoot = [4,1,2]
Output: true
Example 2:


Input: root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2]
Output: false
 

Constraints:

The number of nodes in the root tree is in the range [1, 2000].
The number of nodes in the subRoot tree is in the range [1, 1000].
-104 <= root.val <= 104
-104 <= subRoot.val <= 104
</pre>

Solution:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if root is None and subRoot is None:
            return True
        if root is None or subRoot is None:
            return False
        res = False
        if root.val == subRoot.val:
            left_check = self.matchSubtree(root.left, subRoot.left)
            right_check = self.matchSubtree(root.right, subRoot.right)
            res = (left_check and right_check)
            if res == True:
                return (left_check and right_check)
        return (self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot))

    def matchSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if root is None and subRoot is None:
            return True
        if root is None or subRoot is None:
            return False
        res = False
        if root.val == subRoot.val:
            left_check = self.matchSubtree(root.left, subRoot.left)
            right_check = self.matchSubtree(root.right, subRoot.right)
            res = (left_check and right_check)
        return res
```


### 29. [Leetcode 235](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/) : Lowest Common Ancestor of a Binary Search Tree

<pre>
Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

 

Example 1:


Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.
Example 2:


Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
Example 3:

Input: root = [2,1], p = 2, q = 1
Output: 2
 

Constraints:

The number of nodes in the tree is in the range [2, 105].
-109 <= Node.val <= 109
All Node.val are unique.
p != q
p and q will exist in the BST.
</pre>

Solution:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root is None:
            return root
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        return root
```


### 30. [Leetcode 102](https://leetcode.com/problems/binary-tree-level-order-traversal/description/) : Binary Tree Level Order Traversal

<pre>
Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

 

Example 1:


Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]
Example 2:

Input: root = [1]
Output: [[1]]
Example 3:

Input: root = []
Output: []
 

Constraints:

The number of nodes in the tree is in the range [0, 2000].
-1000 <= Node.val <= 1000
</pre>

Solution:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        res = list()
        if root is None:
            return res
        _queue = list()
        _queue.append(root)
        i = 0
        while i < len(_queue):
            _list = list()
            _len = len(_queue)
            while i < _len:
                node = _queue[i]
                _list.append(node.val)
                if node.left:
                    _queue.append(node.left)
                if node.right:
                    _queue.append(node.right)
                i += 1
            res.append(_list)
        return res
```


### 31. [Leetcode 98](https://leetcode.com/problems/validate-binary-search-tree/description/) : Validate Binary Search Tree

<pre>
Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

The left 
subtree
 of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
 

Example 1:


Input: root = [2,1,3]
Output: true
Example 2:


Input: root = [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.
 

Constraints:

The number of nodes in the tree is in the range [1, 104].
-231 <= Node.val <= 231 - 1
</pre>

Solution:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def valid(root: Optional[TreeNode], left: float, right: float):
            if not root:
                return True
            if not (root.val > left and root.val < right):
                return False
            return (valid(root.left, left, root.val) and valid(root.right, root.val, right))
        return valid(root, float("-inf"), float("+inf"))
```


### 32. [Leetcode 230](https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/) : Kth Smallest Element in a BST

<pre>
Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.

 

Example 1:


Input: root = [3,1,4,null,2], k = 1
Output: 1
Example 2:


Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3
 

Constraints:

The number of nodes in the tree is n.
1 <= k <= n <= 104
0 <= Node.val <= 104
 

Follow up: If the BST is modified often (i.e., we can do insert and delete operations) and you need to find the kth smallest frequently, how would you optimize?
</pre>

Solution:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        n = 0
        stack = []
        curr = root
        while curr or stack:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            n += 1
            if n == k:
                return curr.val
            curr = curr.right
```


### 33. [Leetcode 105](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/) : Construct Binary Tree from Preorder and Inorder Traversal

<pre>
Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

 

Example 1:


Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
Example 2:

Input: preorder = [-1], inorder = [-1]
Output: [-1]
 

Constraints:

1 <= preorder.length <= 3000
inorder.length == preorder.length
-3000 <= preorder[i], inorder[i] <= 3000
preorder and inorder consist of unique values.
Each value of inorder also appears in preorder.
preorder is guaranteed to be the preorder traversal of the tree.
inorder is guaranteed to be the inorder traversal of the tree.
</pre>

Solution:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None
        root = TreeNode(preorder[0])
        mid = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1:], inorder[mid + 1:])
        return root
```


### 34. [Leetcode 124](https://leetcode.com/problems/binary-tree-maximum-path-sum/description/) : Binary Tree Maximum Path Sum

<pre>
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any non-empty path.

 

Example 1:


Input: root = [1,2,3]
Output: 6
Explanation: The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.
Example 2:


Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.
 

Constraints:

The number of nodes in the tree is in the range [1, 3 * 104].
-1000 <= Node.val <= 1000
</pre>

Solution:

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        res = [root.val]
        def evaluate(root):
            if not root:
                return 0
            leftMax = evaluate(root.left)
            rightMax = evaluate(root.right)

            leftMax = max(leftMax, 0)
            rightMax = max(rightMax, 0)

            res[0] = max(res[0], root.val + leftMax + rightMax)

            return (root.val + max(leftMax, rightMax))
        evaluate(root)
        return res[0]
```


### 35. [Leetcode 297](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/description/) : Serialize and Deserialize Binary Tree

<pre>
Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

 

Example 1:


Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]
Example 2:

Input: root = []
Output: []
 

Constraints:

The number of nodes in the tree is in the range [0, 104].
-1000 <= Node.val <= 1000
</pre>

Solution:

```python
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        res = []
        def dfs(node):
            if not node:
                res.append("N")
                return
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)
            return
        dfs(root)
        return ",".join(res)
        

    def deserialize(self, data):
        vals = data.split(",")
        self.i = 0
        def dfs():
            if vals[self.i] == "N":
                self.i += 1
                return None
            node = TreeNode(int(vals[self.i]))
            self.i += 1
            node.left = dfs()
            node.right = dfs()
            return node
        return dfs()
        

# Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))
```

## Heap and Priority Queue

### 36. [Leetcode 295](https://leetcode.com/problems/find-median-from-data-stream/description/) : Find Median from Data Stream

<pre>
The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.

For example, for arr = [2,3,4], the median is 3.
For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
Implement the MedianFinder class:

MedianFinder() initializes the MedianFinder object.
void addNum(int num) adds the integer num from the data stream to the data structure.
double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.
 

Example 1:

Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]

Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
 

Constraints:

-105 <= num <= 105
There will be at least one element in the data structure before calling findMedian.
At most 5 * 104 calls will be made to addNum and findMedian.
 

Follow up:

If all integer numbers from the stream are in the range [0, 100], how would you optimize your solution?
If 99% of all integer numbers from the stream are in the range [0, 100], how would you optimize your solution?
</pre>

Solution:

```python
class MedianFinder:

    def __init__(self):
        self.left_max_heap, self.right_min_heap = [], [] # left will be max heap, right is min heap
        

    def addNum(self, num: int) -> None:
        heapq.heappush(self.left_max_heap, num * -1)
        if len(self.right_min_heap) > 0 and self.left_max_heap[0] * -1 > self.right_min_heap[0]:
            ele = heapq.heappop(self.left_max_heap) * -1
            heapq.heappush(self.right_min_heap, ele)
        if len(self.left_max_heap) > len(self.right_min_heap) + 1:
            ele = heapq.heappop(self.left_max_heap) * -1
            heapq.heappush(self.right_min_heap, ele)
        if len(self.left_max_heap) + 1 < len(self.right_min_heap):
            ele = heapq.heappop(self.right_min_heap)
            heapq.heappush(self.left_max_heap, ele * -1)

    def findMedian(self) -> float:
        if len(self.left_max_heap) > len(self.right_min_heap):
            return self.left_max_heap[0] * -1
        if len(self.left_max_heap) < len(self.right_min_heap):
            return self.right_min_heap[0]
        return ((self.left_max_heap[0] * -1) + self.right_min_heap[0]) / 2
        


# Your MedianFinder object will be instantiated and called as such:
# obj = MedianFinder()
# obj.addNum(num)
# param_2 = obj.findMedian()
```

## Backtracking

### 37. [Leetcode 39](https://leetcode.com/problems/combination-sum/description/) : Combination Sum

<pre>
Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the 
frequency
 of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

 

Example 1:

Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
Example 2:

Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]
Example 3:

Input: candidates = [2], target = 1
Output: []
 

Constraints:

1 <= candidates.length <= 30
2 <= candidates[i] <= 40
All elements of candidates are distinct.
1 <= target <= 40
</pre>

Solution:

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        def dfs(i, combination, count):
            if count == target:
                res.append(combination.copy())
                return
            if i > len(candidates) - 1 or count > target:
                return
            ele = candidates[i]
            combination.append(ele)
            dfs(i, combination, count + ele)
            combination.pop()
            dfs(i + 1, combination, count)
        dfs(0, [], 0)
        return res
```


### 38. [Leetcode 79](https://leetcode.com/problems/word-search/description/) : Word Search

<pre>
Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

 

Example 1:


Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
Example 2:


Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true
Example 3:


Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
Output: false
 

Constraints:

m == board.length
n = board[i].length
1 <= m, n <= 6
1 <= word.length <= 15
board and word consists of only lowercase and uppercase English letters.
 

Follow up: Could you use search pruning to make your solution faster with a larger board?
</pre>

Solution:

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        ROWS, COLS = len(board), len(board[0])
        visited = set()
        def dfs(r, c, i):
            if i == len(word):
                return True
            if (r < 0 or 
                c < 0 or 
                r >= ROWS or 
                c >= COLS or
                (r, c) in visited or
                board[r][c] != word[i]):
                return False
            visited.add((r,c))
            res = ( dfs(r+1, c, i+1) or
                    dfs(r-1, c, i+1) or
                    dfs(r, c+1, i+1) or
                    dfs(r, c-1, i+1)
                    )
            visited.remove((r,c))
            return res
        for R in range(ROWS):
            for C in range(COLS):
                if dfs(R, C, 0): return True
        return False
```

## Tries

### 39. [Leetcode 208](https://leetcode.com/problems/implement-trie-prefix-tree/description/) : Implement Trie (Prefix Tree)

<pre>
A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

Trie() Initializes the trie object.
void insert(String word) Inserts the string word into the trie.
boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
 

Example 1:

Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]

Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True
 

Constraints:

1 <= word.length, prefix.length <= 2000
word and prefix consist only of lowercase English letters.
At most 3 * 104 calls in total will be made to insert, search, and startsWith.
</pre>

Solution:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class Trie:

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True
        
    def search(self, word: str) -> bool:
        cur = self.root
        for c in word:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return cur.endOfWord
        

    def startsWith(self, prefix: str) -> bool:
        cur = self.root
        for c in prefix:
            if c not in cur.children:
                return False
            cur = cur.children[c]
        return True    


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)
```


### 40. [Leetcode 211](https://leetcode.com/problems/design-add-and-search-words-data-structure/description/) : Design Add and Search Words Data Structure

<pre>
Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the WordDictionary class:

WordDictionary() Initializes the object.
void addWord(word) Adds word to the data structure, it can be matched later.
bool search(word) Returns true if there is any string in the data structure that matches word or false otherwise. word may contain dots '.' where dots can be matched with any letter.
 

Example:

Input
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
Output
[null,null,null,null,false,true,true,true]

Explanation
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); // return False
wordDictionary.search("bad"); // return True
wordDictionary.search(".ad"); // return True
wordDictionary.search("b.."); // return True
 

Constraints:

1 <= word.length <= 25
word in addWord consists of lowercase English letters.
word in search consist of '.' or lowercase English letters.
There will be at most 2 dots in word for search queries.
At most 104 calls will be made to addWord and search.
</pre>

Solution:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False

class WordDictionary:

    def __init__(self):
        self.root = TrieNode()
        
    def addWord(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True
        

    def search(self, word: str) -> bool:
        def dfs(j, root):
            cur = root
            for i in range(j, len(word)):
                c = word[i]
                if c == '.':
                    for child in cur.children.values():
                        if dfs(i+1, child):
                            return True
                    return False
                else:
                    if c not in cur.children:
                        return False
                    cur = cur.children[c]
            return cur.endOfWord
        return dfs(0, self.root)
        


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)
```


### 41. [Leetcode 212](https://leetcode.com/problems/word-search-ii/description/) : Word Search II

<pre>
Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

 

Example 1:


Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
Example 2:


Input: board = [["a","b"],["c","d"]], words = ["abcb"]
Output: []
 

Constraints:

m == board.length
n == board[i].length
1 <= m, n <= 12
board[i][j] is a lowercase English letter.
1 <= words.length <= 3 * 104
1 <= words[i].length <= 10
words[i] consists of lowercase English letters.
All the strings of words are unique.
</pre>

Solution:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endOfWord = False
    def add(self, word):
        cur = self
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.endOfWord = True

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()
        rows, cols = len(board), len(board[0])
        for word in words:
            root.add(word)
        res, visited = set(), set()
        def dfs(r, c, node, word):
            if (r < 0 or r >= rows or c < 0 or c >= cols or ((r,c) in visited) or (board[r][c] not in node.children)):
                return
            visited.add((r,c))
            node = node.children[board[r][c]]
            word += board[r][c]
            if node.endOfWord == True:
                res.add(word)
            dfs(r + 1, c, node, word)
            dfs(r - 1, c, node, word)
            dfs(r, c + 1, node, word)
            dfs(r, c - 1, node, word)
            visited.remove((r,c))
        for r in range(rows):
            for c in range(cols):
                dfs(r, c, root, "")
        return list(res)
```

## Graphs


### 42. [Leetcode 200](https://leetcode.com/problems/number-of-islands/description/) : Number of Islands

<pre>
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

 

Example 1:

Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
Example 2:

Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
 

Constraints:

m == grid.length
n == grid[i].length
1 <= m, n <= 300
grid[i][j] is '0' or '1'.
</pre>

Solution:

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        rows, cols = len(grid), len(grid[0])
        visited = set()
        res = 0
        def dfs(r,c):
            if not (r in range(rows) and
                c in range(cols) and
                (r,c) not in visited and
                grid[r][c] == '1'):
                return
            visited.add((r,c))
            dfs(r+1, c)
            dfs(r, c+1)
            dfs(r-1,c)
            dfs(r, c-1)
            
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == '1' and (r,c) not in visited:
                    res += 1
                    dfs(r,c)
        return res
```


### 43. [Leetcode 133](https://leetcode.com/problems/clone-graph/description/) : Clone Graph

<pre>
Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}
 

Test case format:

For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with val == 1, the second node with val == 2, and so on. The graph is represented in the test case using an adjacency list.

An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.

The given node will always be the first node with val = 1. You must return the copy of the given node as a reference to the cloned graph.

 

Example 1:


Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.
1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
Example 2:


Input: adjList = [[]]
Output: [[]]
Explanation: Note that the input contains one empty list. The graph consists of only one node with val = 1 and it does not have any neighbors.
Example 3:

Input: adjList = []
Output: []
Explanation: This an empty graph, it does not have any nodes.
 

Constraints:

The number of nodes in the graph is in the range [0, 100].
1 <= Node.val <= 100
Node.val is unique for each node.
There are no repeated edges and no self-loops in the graph.
The Graph is connected and all nodes can be visited starting from the given node.
</pre>

Solution:

```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if node is None:
            return None
        copy = {}
        def clone(node):
            if node in copy:
                return copy[node]
            _clone = Node(node.val)
            copy[node] = _clone
            for ele in node.neighbors:
                _clone.neighbors.append(clone(ele))
            return _clone
        return clone(node)
```


### 44. [Leetcode 417](https://leetcode.com/problems/pacific-atlantic-water-flow/description/) : Pacific Atlantic Water Flow

<pre>
There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean. The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches the island's right and bottom edges.

The island is partitioned into a grid of square cells. You are given an m x n integer matrix heights where heights[r][c] represents the height above sea level of the cell at coordinate (r, c).

The island receives a lot of rain, and the rain water can flow to neighboring cells directly north, south, east, and west if the neighboring cell's height is less than or equal to the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.

Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.

 

Example 1:


Input: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
Explanation: The following cells can flow to the Pacific and Atlantic oceans, as shown below:
[0,4]: [0,4] -> Pacific Ocean 
       [0,4] -> Atlantic Ocean
[1,3]: [1,3] -> [0,3] -> Pacific Ocean 
       [1,3] -> [1,4] -> Atlantic Ocean
[1,4]: [1,4] -> [1,3] -> [0,3] -> Pacific Ocean 
       [1,4] -> Atlantic Ocean
[2,2]: [2,2] -> [1,2] -> [0,2] -> Pacific Ocean 
       [2,2] -> [2,3] -> [2,4] -> Atlantic Ocean
[3,0]: [3,0] -> Pacific Ocean 
       [3,0] -> [4,0] -> Atlantic Ocean
[3,1]: [3,1] -> [3,0] -> Pacific Ocean 
       [3,1] -> [4,1] -> Atlantic Ocean
[4,0]: [4,0] -> Pacific Ocean 
       [4,0] -> Atlantic Ocean
Note that there are other possible paths for these cells to flow to the Pacific and Atlantic oceans.
Example 2:

Input: heights = [[1]]
Output: [[0,0]]
Explanation: The water can flow from the only cell to the Pacific and Atlantic oceans.
 

Constraints:

m == heights.length
n == heights[r].length
1 <= m, n <= 200
0 <= heights[r][c] <= 105
</pre>

Solution:

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        rows, cols = len(heights), len(heights[0])
        pac, atl = set(), set()
        res = []
        def dfs(r, c, visited, parent):
            if (r < 0 or r >= rows or c < 0 or c >= cols or (r,c) in visited or heights[r][c] < parent):
                return
            visited.add((r,c))
            dfs(r + 1, c, visited, heights[r][c])
            dfs(r - 1, c, visited, heights[r][c])
            dfs(r, c + 1, visited, heights[r][c])
            dfs(r, c - 1, visited, heights[r][c])
        for c in range(cols):
            dfs(0, c, pac, heights[0][c])
            dfs(rows-1, c, atl, heights[rows-1][c])
        for r in range(rows):
            dfs(r, 0, pac, heights[r][0])
            dfs(r, cols-1, atl, heights[r][cols-1])
        for r in range(rows):
            for c in range(cols):
                if (r,c) in pac and (r,c) in atl:
                    res.append([r,c])
        return res
```

### 45. [Leetcode 207](https://leetcode.com/problems/course-schedule/description/) : Course Schedule

<pre>
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.

 

Example 1:

Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0. So it is possible.
Example 2:

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
 

Constraints:

1 <= numCourses <= 2000
0 <= prerequisites.length <= 5000
prerequisites[i].length == 2
0 <= ai, bi < numCourses
All the pairs prerequisites[i] are unique.
</pre>

Solution:

```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        if len(prerequisites) == 0:
            return True
        preMap = {}
        visited = set()
        res = True
        for i in range(len(prerequisites)):
            [course, prerequisite] = prerequisites[i]
            if course not in preMap:
                preMap[course] = []
            preMap[course].append(prerequisite)
        def dfs(course):
            if course in visited:
                return False
            if course not in preMap:
                return True
            visited.add(course)
            for pre in preMap[course]:
                if not dfs(pre):
                    return False
            visited.remove(course)
            del preMap[course]
            return res
        for course in range(numCourses):
            if not dfs(course):
                return False
        return True
```


### 46. [Leetcode Premium](https://leetcode.com/problems/graph-valid-tree/description/) : Graph Valid Tree

<pre>
Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.

Example 1:

Input:
n = 5
edges = [[0, 1], [0, 2], [0, 3], [1, 4]]

Output:
true
Example 2:

Input:
n = 5
edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]

Output:
false
Note:

You can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.
Constraints:

1 <= n <= 100
0 <= edges.length <= n * (n - 1) / 2
</pre>

Solution:

```python
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if not n:
            return True
        childToParentMap = {i: [] for i in range(n)}
        for [parent, child] in edges:
            childToParentMap[child].append(parent)
            childToParentMap[parent].append(child)

        visited = set()
        def dfs(node, prev):
            if node in visited:
                return False
            visited.add(node)
            for val in childToParentMap[node]:
                if val == prev:
                    continue
                if not dfs(val, node):
                    return False
            return True

        return (dfs(0, -1) and len(visited) == n)
```


### 47. [Leetcode Premium](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/description/) : Number of Connected Components in an Undirected Graph

<pre>
There is an undirected graph with n nodes. There is also an edges array, where edges[i] = [a, b] means that there is an edge between node a and node b in the graph.

The nodes are numbered from 0 to n - 1.

Return the total number of connected components in that graph.

Example 1:

Input:
n=3
edges=[[0,1], [0,2]]

Output:
1
Example 2:

Input:
n=6
edges=[[0,1], [1,2], [2,3], [4,5]]

Output:
2
Constraints:

1 <= n <= 100
0 <= edges.length <= n * (n - 1) / 2
</pre>

Solution:

```python
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        parents = [i for i in range(n)]
        rank = [1] * n
        parentSet = set()
        def findParent(n):
            res = n
            if res == parents[res]:
                return res
            parents[res] = findParent(parents[res])
            return parents[res]
        def union(n1, n2):
            p1, p2 = findParent(n1), findParent(n2)
            if p1 == p2:
                return 0
            if rank[p1] > rank[p2]:
                parents[p2] = p1
                rank[p1] += rank[p2]
            else:
                parents[p1] = p2
                rank[p2] += rank[p1]
            return 1
        res = n
        for [n1, n2] in edges:
            res -= union(n1, n2)
        return res
```

## Advanced Graphs

### 48. [Leetcode Premium](https://leetcode.com/problems/alien-dictionary/description/) : Alien Dictionary

<pre>
There is a foreign language which uses the latin alphabet, but the order among letters is not "a", "b", "c" ... "z" as in English.

You receive a list of non-empty strings words from the dictionary, where the words are sorted lexicographically based on the rules of this new language.

Derive the order of letters in this language. If the order is invalid, return an empty string. If there are multiple valid order of letters, return any of them.

A string a is lexicographically smaller than a string b if either of the following is true:

The first letter where they differ is smaller in a than in b.
There is no index i such that a[i] != b[i] and a.length < b.length.
Example 1:

Input: ["z","o"]

Output: "zo"
Explanation:
From "z" and "o", we know 'z' < 'o', so return "zo".

Example 2:

Input: ["hrn","hrf","er","enn","rfnn"]

Output: "hernf"
Explanation:

from "hrn" and "hrf", we know 'n' < 'f'
from "hrf" and "er", we know 'h' < 'e'
from "er" and "enn", we know get 'r' < 'n'
from "enn" and "rfnn" we know 'e'<'r'
so one possibile solution is "hernf"
Constraints:

The input words will contain characters only from lowercase 'a' to 'z'.
1 <= words.length <= 100
1 <= words[i].length <= 100
</pre>

Solution:

```python
class Solution:
    def foreignDictionary(self, words: List[str]) -> str:
        adj = { c : set() for word in words for c in word}
        for i in range(len(words) - 1):
            s1, s2 = words[i], words[i+1]
            minLen = min(len(s1), len(s2))
            if len(s1) > len(s2) and s1[:minLen] == s2[:minLen]:
                return ""
            for i in range(minLen):
                if s1[i] != s2[i]:
                    adj[s1[i]].add(s2[i])
                    break
        visited = {} # False not cyclic, True for cyclic
        res = []
        def dfs(node):
            if node in visited:
                return visited[node]
            visited[node] = True
            for child in adj[node]:
                if dfs(child):
                    return True
            visited[node] = False
            res.append(node)
            return False
        for k in adj:
            if dfs(k):
                return ""
        res.reverse()
        return ''.join(res)
```

## Dynamic Programming (1-D)

### 49. [Leetcode 70](https://leetcode.com/problems/climbing-stairs/description/) : Climbing Stairs

<pre>
You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

 

Example 1:

Input: n = 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
Example 2:

Input: n = 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step
 

Constraints:

1 <= n <= 45
</pre>

Solution:

```python
# Recursive solution 1

# class Solution:
#     def climbStairs(self, n: int) -> int:
#         memo = {}
#         def dfs(steps):
#             if steps < 3:
#                 return steps
#             if steps in memo:
#                 return memo[steps]
#             res = dfs(steps - 2) + dfs(steps - 1)
#             memo[steps] = res
#             return res
#         return dfs(n)

# Recursive solution 2

class Solution:
    def climbStairs(self, n: int) -> int:
        memo = {}
        def dfs(steps):
            if steps <= 0:
                return 0
            if steps in memo:
                return memo[steps]
            res = dfs(steps - 2) + dfs(steps - 1)
            memo[steps] = res
            return res
        return dfs(n)

# Iterative solution

# class Solution:
#     def climbStairs(self, n: int) -> int:
#         step1, step2 = 1, 1
#         for i in range(n - 1):
#             temp = step1 + step2
#             step1 = step2
#             step2 = temp
#         return step2
```


### 50. [Leetcode 198](https://leetcode.com/problems/house-robber/description/) : House Robber

<pre>
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
Example 2:

Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.
 

Constraints:

1 <= nums.length <= 100
0 <= nums[i] <= 400
</pre>

Solution:

```python
# Recursive solution 1

# class Solution:
#     def rob(self, nums: List[int]) -> int:
#         memo = {}
#         def dfs(nums, index):
#             if (index == len(nums) -1):
#                 return nums[index]
#             if (index == len(nums) - 2):
#                 return max(nums[index], nums[index+1])
#             if (index == len(nums) - 3):
#                 return nums[index] + nums[index+2]
#             if index in memo:
#                 return memo[index]
#             memo[index] = nums[index] + max(dfs(nums, index+2), dfs(nums, index+3))
#             return memo[index]
#         res = nums[0]
#         for i in range(len(nums)):
#             res = max(res, dfs(nums, i))
#         return res

# Recursive solution 2

# class Solution:
#     def rob(self, nums: List[int]) -> int:
#         memo = {}
#         def dfs(nums, index):
#             if (index == len(nums) -1):
#                 return nums[index]
#             if (index == len(nums) - 2):
#                 return max(nums[index], nums[index+1])
#             if (index == len(nums) - 3):
#                 return nums[index] + nums[index+2]
#             if index in memo:
#                 return memo[index]
#             memo[index] = nums[index] + max(dfs(nums, index+2), dfs(nums, index+3))
#             return memo[index]
#         res = nums[0]
#         for i in range(len(nums)):
#             res = max(res, dfs(nums, i))
#         return res

# Iterative solution

class Solution:
    def rob(self, nums: List[int]) -> int:
        rob1, rob2 = 0, 0
        for num in nums:
            temp = max(rob1 + num, rob2)
            rob1 = rob2
            rob2 = temp
        return rob2
```


### 51. [Leetcode 213](https://leetcode.com/problems/house-robber-ii/description/) : House Robber II

<pre>
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

 

Example 1:

Input: nums = [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.
Example 2:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
Example 3:

Input: nums = [1,2,3]
Output: 3
 

Constraints:

1 <= nums.length <= 100
0 <= nums[i] <= 1000
</pre>

Solution:

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        return max(nums[0], self.helper(nums[1:]), self.helper(nums[:-1]))

    def helper(self, nums):
        rob1, rob2 = 0, 0
        for n in nums:
            temp = max(rob1 + n, rob2)
            rob1 = rob2
            rob2 = temp
        return rob2
```


### 52. [Leetcode 5](https://leetcode.com/problems/longest-palindromic-substring/description/) : Longest Palindromic Substring

<pre>
Given a string s, return the longest 
palindromic
 
substring
 in s.

 

Example 1:

Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
Example 2:

Input: s = "cbbd"
Output: "bb"
 

Constraints:

1 <= s.length <= 1000
s consist of only digits and English letters.
</pre>

Solution:

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        res, resLen = "", 0
        for i in range(len(s)):
            # for odd string
            l, r = i, i
            while (l > -1 and r < len(s)) and (s[l] == s[r]):
                if (r - l + 1) > resLen:
                    res = s[l:r+1]
                    resLen = r - l + 1
                l -= 1
                r += 1
            # for even string
            l, r = i, i + 1
            while (l > -1 and r < len(s)) and (s[l] == s[r]):
                if (r - l + 1) > resLen:
                    res = s[l:r+1]
                    resLen = r - l + 1
                l -= 1
                r += 1
        return res
```

### 53. [Leetcode 647](https://leetcode.com/problems/palindromic-substrings/description/) : Palindromic Substrings

<pre>
Given a string s, return the number of palindromic substrings in it.

A string is a palindrome when it reads the same backward as forward.

A substring is a contiguous sequence of characters within the string.

 

Example 1:

Input: s = "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".
Example 2:

Input: s = "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
 

Constraints:

1 <= s.length <= 1000
s consists of lowercase English letters.
</pre>

Solution:

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        res = 0
        for i in range(len(s)):
            # Odd substring
            l, r = i, i
            while (l > -1 and r < len(s)) and s[l] == s[r]:
                res += 1
                l -= 1
                r += 1
            # Even substring
            l, r = i, i+1
            while (l > -1 and r < len(s)) and s[l] == s[r]:
                res += 1
                l -= 1
                r += 1
        return res
```


### 54. [Leetcode 91](https://leetcode.com/problems/decode-ways/description/) : Decode Ways

<pre>
You have intercepted a secret message encoded as a string of numbers. The message is decoded via the following mapping:

"1" -> 'A'

"2" -> 'B'

...

"25" -> 'Y'

"26" -> 'Z'

However, while decoding the message, you realize that there are many different ways you can decode the message because some codes are contained in other codes ("2" and "5" vs "25").

For example, "11106" can be decoded into:

"AAJF" with the grouping (1, 1, 10, 6)
"KJF" with the grouping (11, 10, 6)
The grouping (1, 11, 06) is invalid because "06" is not a valid code (only "6" is valid).
Note: there may be strings that are impossible to decode.

Given a string s containing only digits, return the number of ways to decode it. If the entire string cannot be decoded in any valid way, return 0.

The test cases are generated so that the answer fits in a 32-bit integer.

 

Example 1:

Input: s = "12"

Output: 2

Explanation:

"12" could be decoded as "AB" (1 2) or "L" (12).

Example 2:

Input: s = "226"

Output: 3

Explanation:

"226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).

Example 3:

Input: s = "06"

Output: 0

Explanation:

"06" cannot be mapped to "F" because of the leading zero ("6" is different from "06"). In this case, the string is not a valid encoding, so return 0.

 

Constraints:

1 <= s.length <= 100
s contains only digits and may contain leading zero(s).
</pre>

Solution:

```python
# Iterative and space optimized solution

# class Solution:
#     def numDecodings(self, s: str) -> int:
#         dp, dp1 = 0, 1
#         dp2 = 0
#         for i in range(len(s)-1, -1, -1):
#             if s[i] == "0":
#                 dp = 0
#             else:
#                 dp = dp1
#             if i+1 < len(s) and int(s[i:i+2]) > 9 and int(s[i:i+2]) < 27:
#                 dp += dp2
#             dp2 = dp1
#             dp1 = dp
#             dp = 0
#         return dp1


# Recursive solution

class Solution:
    def numDecodings(self, s: str) -> int:
        memo = {}
        def helper(index):
            if index > len(s) - 1:
                return 1
            if s[index] == "0":
                return 0
            if index in memo:
                return memo[index]
            res = helper(index + 1)
            if (index+1 < len(s)) and int(s[index:index+2]) < 27:
                res = res + helper(index + 2)
            memo[index] = res
            return res
        return helper(0)
```


### 55. [Leetcode 322](https://leetcode.com/problems/coin-change/description/) : Coin Change

<pre>
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

 

Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
Example 2:

Input: coins = [2], amount = 3
Output: -1
Example 3:

Input: coins = [1], amount = 0
Output: 0
 

Constraints:

1 <= coins.length <= 12
1 <= coins[i] <= 231 - 1
0 <= amount <= 104
</pre>

Solution:

```python
# Recursive solution (Top Down)

# class Solution:
#     def coinChange(self, coins: List[int], amount: int) -> int:
#         memo = {}
#         def helper(_sum):
#             if _sum == 0:
#                 return 0
#             if _sum in memo:
#                 return memo[_sum]
#             else:
#                 memo[_sum] = amount + 1
#                 for coin in coins:
#                     if _sum - coin > -1:
#                         memo[_sum] = min(memo[_sum], 1 + helper(_sum - coin))
#                 return memo[_sum]
#         res = helper(amount)
#         return res if res != amount + 1 else -1

# Iterative Bottoms Up

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0
        for _amount in range (1, amount + 1):
            for coin in coins:
                if _amount - coin > -1:
                    dp[_amount] = min(dp[_amount], 1 + dp[_amount - coin])
        return dp[amount] if dp[amount] != amount + 1 else -1
```


### 56. [Leetcode 152](https://leetcode.com/problems/maximum-product-subarray/description/) : Maximum Product Subarray

<pre>
Given an integer array nums, find a 
subarray
 that has the largest product, and return the product.

The test cases are generated so that the answer will fit in a 32-bit integer.

 

Example 1:

Input: nums = [2,3,-2,4]
Output: 6
Explanation: [2,3] has the largest product 6.
Example 2:

Input: nums = [-2,0,-1]
Output: 0
Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
 

Constraints:

1 <= nums.length <= 2 * 104
-10 <= nums[i] <= 10
The product of any subarray of nums is guaranteed to fit in a 32-bit integer.
</pre>

Solution:

```python
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        res = max(nums)
        _min, _max = 1, 1
        for num in nums:
            temp = _max * num
            _max = max(_min * num, temp, num)
            _min = min(_min * num, temp, num)
            res = max(_max, res)
        return res
```


### 57. [Leetcode 139](https://leetcode.com/problems/word-break/description/) : Word Break

<pre>
Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.

Note that the same word in the dictionary may be reused multiple times in the segmentation.

 

Example 1:

Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
Example 2:

Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.
Example 3:

Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false
 

Constraints:

1 <= s.length <= 300
1 <= wordDict.length <= 1000
1 <= wordDict[i].length <= 20
s and wordDict[i] consist of only lowercase English letters.
All the strings of wordDict are unique.
</pre>

Solution:

```python
# Recursive solution

# class Solution:
#     def wordBreak(self, s: str, wordDict: List[str]) -> bool:
#         memo = {}
#         def helper(index):
#             if index == len(s):
#                 return True
#             if index in memo:
#                 return memo[index]
#             memo[index] = False
#             for word in wordDict:
#                 if (index + len(word)) <= len(s) and s[index:(index + len(word))] == word:
#                     res = helper(index + len(word))
#                     if res:
#                         memo[index] = res
#                         break
#             return memo[index]
#         return helper(0)

# Iterative solution

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        _lenS = len(s)
        memo = [False] * (_lenS + 1)
        memo[_lenS] = True
        for i in range(_lenS, -1, -1):
            for word in wordDict:
                if (i + len(word)) <= _lenS and s[i: i + len(word)] == word:
                    memo[i] = memo[i + len(word)]
                    if memo[i]:
                        break
        return memo[0]
```


### 58. [Leetcode 300](https://leetcode.com/problems/longest-increasing-subsequence/description/) : Longest Increasing Subsequence

<pre>
Given an integer array nums, return the length of the longest strictly increasing 
subsequence
.

 

Example 1:

Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
Example 2:

Input: nums = [0,1,0,3,2,3]
Output: 4
Example 3:

Input: nums = [7,7,7,7,7,7,7]
Output: 1
 

Constraints:

1 <= nums.length <= 2500
-104 <= nums[i] <= 104
 

Follow up: Can you come up with an algorithm that runs in O(n log(n)) time complexity?
</pre>

Solution:

```python
# Recursive solution

# class Solution:
#     def lengthOfLIS(self, nums: List[int]) -> int:
#         memo = {}
#         def helper(index):
#             if index == len(nums):
#                 return 0
#             if index in memo:
#                 return memo[index]
#             res = 1
#             for i in range(index+1, len(nums)):
#                 if nums[index] < nums[i]:
#                     res = max(res, 1 + helper(i))
#             memo[index] = res
#             return res
#         res = 0
#         for i in range(len(nums)):
#             res = max(res, helper(i))
#         return res

# Iterative solution
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        memo = [1] * (len(nums))
        for i in range(len(nums)-1, -1, -1):
            for j in range(i+1, len(nums)):
                if nums[i] < nums[j]:
                    memo[i] = max(memo[i], 1 + memo[j])
        return max(memo)
```

## Dynamic Programming (2-D)

### 59. [Leetcode 62](https://leetcode.com/problems/unique-paths/description/) : Unique Paths

<pre>
There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The test cases are generated so that the answer will be less than or equal to 2 * 109.

 

Example 1:


Input: m = 3, n = 7
Output: 28
Example 2:

Input: m = 3, n = 2
Output: 3
Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Down -> Down
2. Down -> Down -> Right
3. Down -> Right -> Down
 

Constraints:

1 <= m, n <= 100
</pre>

Solution:

```python
# Recursive solution

# class Solution:
#     def uniquePaths(self, m: int, n: int) -> int:
#         memo = {}
#         def helper(r, c):
#             if r == m or c == n:
#                 return 0
#             if r == m-1 or c == n-1:
#                 return 1
#             if (r,c) in memo:
#                 return memo[(r,c)]
#             res = helper(r+1, c) + helper(r, c+1)
#             memo[(r,c)] = res
#             return res
#         return helper(0,0)


# Bottom's up

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = 1
        memo = [0] * n
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                memo[j] = dp + memo[j]
                dp = memo[j]
            dp = 0
        return memo[0]
```

### 60. [Leetcode 1143](https://leetcode.com/problems/longest-common-subsequence/description/) : Longest Common Subsequence

<pre>
Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".
A common subsequence of two strings is a subsequence that is common to both strings.

 

Example 1:

Input: text1 = "abcde", text2 = "ace" 
Output: 3  
Explanation: The longest common subsequence is "ace" and its length is 3.
Example 2:

Input: text1 = "abc", text2 = "abc"
Output: 3
Explanation: The longest common subsequence is "abc" and its length is 3.
Example 3:

Input: text1 = "abc", text2 = "def"
Output: 0
Explanation: There is no such common subsequence, so the result is 0.
 

Constraints:

1 <= text1.length, text2.length <= 1000
text1 and text2 consist of only lowercase English characters.
</pre>

Solution:

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        memo = [[0 for i in range(len(text2) + 1)] for j in range(len(text1) + 1)]
        for i in range(len(text1)-1, -1, -1):
            for j in range(len(text2)-1, -1, -1):
                if text1[i] == text2[j]:
                    memo[i][j] = 1 + memo[i+1][j+1]
                else:
                    memo[i][j] = max(memo[i+1][j], memo[i][j+1])
        return memo[0][0]
```

## Greedy

### 61. [Leetcode 53](https://leetcode.com/problems/maximum-subarray/description/) : Maximum Subarray

<pre>
Given an integer array nums, find the 
subarray
 with the largest sum, and return its sum.

 

Example 1:

Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: The subarray [4,-1,2,1] has the largest sum 6.
Example 2:

Input: nums = [1]
Output: 1
Explanation: The subarray [1] has the largest sum 1.
Example 3:

Input: nums = [5,4,-1,7,8]
Output: 23
Explanation: The subarray [5,4,-1,7,8] has the largest sum 23.
 

Constraints:

1 <= nums.length <= 105
-104 <= nums[i] <= 104
 

Follow up: If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.
</pre>

Solution:

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        res, _sum = nums[0], 0
        for num in nums:
            if _sum < 0:
                _sum = 0
            _sum += num
            res = max(res, _sum)
        return res
```


### 62. [Leetcode 55](https://leetcode.com/problems/jump-game/description/) : Jump Game

<pre>
You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.

 

Example 1:

Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
Example 2:

Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
 

Constraints:

1 <= nums.length <= 104
0 <= nums[i] <= 105
</pre>

Solution:

```python
# DP approach

# class Solution:
#     def canJump(self, nums: List[int]) -> bool:
#         memo = [False] * len(nums)
#         memo[len(nums) - 1] = True
#         for i in range(len(nums)-2, -1, -1):
#             for j in range(nums[i]+1):
#                 if memo[i + j] == True:
#                     memo[i] = True
#                     break
#         return memo[0] 

# Greedy approach

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        _len = len(nums)
        target = _len-1
        res = True
        for i in range(_len-1, -1, -1):
            if nums[i] >= target - i:
                target = i
                res = True
            else:
                res = False
        return res
```

## Intervals

### 63. [Leetcode 57](https://leetcode.com/problems/insert-interval/description/) : Insert Interval

<pre>
You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.

Note that you don't need to modify intervals in-place. You can make a new array and return it.

 

Example 1:

Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
Example 2:

Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
 

Constraints:

0 <= intervals.length <= 104
intervals[i].length == 2
0 <= starti <= endi <= 105
intervals is sorted by starti in ascending order.
newInterval.length == 2
0 <= start <= end <= 105
</pre>

Solution:

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res = []
        for i in range(len(intervals)):
            if newInterval[1] < intervals[i][0]:
                res.append(newInterval)
                return res + intervals[i:]
            elif newInterval[0] > intervals[i][1]:
                res.append(intervals[i])
            else:
                newInterval = [min(intervals[i][0], newInterval[0]), max(intervals[i][1], newInterval[1])]
        res.append(newInterval)
        return res
```


### 64. [Leetcode 56](https://leetcode.com/problems/merge-intervals/description/) : Merge Intervals

<pre>
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

 

Example 1:

Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
Example 2:

Input: intervals = [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.
 

Constraints:

1 <= intervals.length <= 104
intervals[i].length == 2
0 <= starti <= endi <= 104
</pre>

Solution:

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key = lambda i : i[0])
        res = [intervals[0]]
        for i in range(1,len(intervals)):
            interval = intervals[i]
            if interval[0] <= res[-1][1]:
                res[-1][1] = max(interval[1], res[-1][1])
            else:
                res.append(interval)
        return res
```

### 65. [Leetcode 435](https://leetcode.com/problems/non-overlapping-intervals/description/) : Non-overlapping Intervals

<pre>
Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

Note that intervals which only touch at a point are non-overlapping. For example, [1, 2] and [2, 3] are non-overlapping.

 

Example 1:

Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and the rest of the intervals are non-overlapping.
Example 2:

Input: intervals = [[1,2],[1,2],[1,2]]
Output: 2
Explanation: You need to remove two [1,2] to make the rest of the intervals non-overlapping.
Example 3:

Input: intervals = [[1,2],[2,3]]
Output: 0
Explanation: You don't need to remove any of the intervals since they're already non-overlapping.
 

Constraints:

1 <= intervals.length <= 105
intervals[i].length == 2
-5 * 104 <= starti < endi <= 5 * 104
</pre>

Solution:

```python
# class Solution:
#     def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
#         intervals.sort(key = lambda i: i[0])
#         res = [intervals[0]]
#         i = 1
#         _sum = 0
#         while i < len(intervals):
#             if intervals[i][0] < res[-1][1]:
#                 if intervals[i][1] < res[-1][1]:
#                     res[-1] = intervals[i]
#                 _sum += 1
#             else:
#                 res.append(intervals[i])
#             i += 1
#         return _sum

class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key = lambda i: i[0])
        prevEnd = intervals[0][1]
        res = 0
        for start, end in intervals[1:]:
            if start >= prevEnd:
                prevEnd = end
            else:
                res += 1
                prevEnd = min(prevEnd, end)
        return res
```

### 66. [Leetcode Premium](https://leetcode.com/problems/meeting-rooms/description/) : Meeting Rooms

<pre>
Given an array of meeting time interval objects consisting of start and end times [[start_1,end_1],[start_2,end_2],...] (start_i < end_i), determine if a person could add all meetings to their schedule without any conflicts.

Example 1:

Input: intervals = [(0,30),(5,10),(15,20)]

Output: false
Explanation:

(0,30) and (5,10) will conflict
(0,30) and (15,20) will conflict
Example 2:

Input: intervals = [(5,8),(9,15)]

Output: true
Note:

(0,8),(8,10) is not considered a conflict at 8
Constraints:

0 <= intervals.length <= 500
0 <= intervals[i].start < intervals[i].end <= 1,000,000
</pre>

Solution:

```python
"""
Definition of Interval:
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    def canAttendMeetings(self, intervals: List[Interval]) -> bool:
        if len(intervals) == 0:
            return True
        intervals.sort(key = lambda i: i.start)
        lastEndTime = intervals[0].end
        for interval in intervals[1:]:
            if interval.start < lastEndTime:
                return False
            else:
                lastEndTime = interval.end
        return True
```

### 67. [Leetcode Premium](https://leetcode.com/problems/meeting-rooms-ii/description/) : Meeting Rooms II

<pre>
Given an array of meeting time interval objects consisting of start and end times [[start_1,end_1],[start_2,end_2],...] (start_i < end_i), find the minimum number of days required to schedule all meetings without any conflicts.

Example 1:

Input: intervals = [(0,40),(5,10),(15,20)]

Output: 2
Explanation:
day1: (0,40)
day2: (5,10),(15,20)

Example 2:

Input: intervals = [(4,9)]

Output: 1
Note:

(0,8),(8,10) is not considered a conflict at 8
Constraints:

0 <= intervals.length <= 500
0 <= intervals[i].start < intervals[i].end <= 1,000,000

</pre>

Solution:

```python
"""
Definition of Interval:
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    def minMeetingRooms(self, intervals: List[Interval]) -> int:
        start_list = []
        end_list = []
        for interval in intervals:
            start_list.append(interval.start)
            end_list.append(interval.end)
        start_list.sort()
        end_list.sort()
        i, j = 0, 0
        res, concurrent_meetings = 0,0
        while i < len(start_list) and j < len(end_list):
            if start_list[i] < end_list[j]:
                concurrent_meetings += 1
                res = max(res, concurrent_meetings)
                i += 1
            else:
                concurrent_meetings -= 1
                j += 1
        return res
```

## Maths and Geometry

### 68. [Leetcode 48](https://leetcode.com/problems/rotate-image/description/) : Rotate Image

<pre>
You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

 

Example 1:


Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
Example 2:


Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
 

Constraints:

n == matrix.length == matrix[i].length
1 <= n <= 20
-1000 <= matrix[i][j] <= 1000
</pre>

Solution:

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        left, right = 0, len(matrix) - 1
        while left < right:
            for i in range(right - left):
                top, bottom = left, right

                temp = matrix[top][left + i]
                matrix[top][left + i] = matrix[bottom - i][left]
                matrix[bottom - i][left] = matrix[bottom][right - i]
                matrix[bottom][right - i] = matrix[top + i][right]
                matrix[top + i][right] = temp
            left += 1
            right -= 1
        return matrix
```

### 69. [Leetcode 54](https://leetcode.com/problems/spiral-matrix/description/) : Spiral Matrix

<pre>
Given an m x n matrix, return all elements of the matrix in spiral order.

 

Example 1:


Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]
Example 2:


Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
 

Constraints:

m == matrix.length
n == matrix[i].length
1 <= m, n <= 10
-100 <= matrix[i][j] <= 100
</pre>

Solution:

```python
# class Solution:
#     def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
#         left, top = 0, 0
#         right, bottom = len(matrix[0]), len(matrix)
#         res = []
#         while left < right and top < bottom:
#             for i in range(left, right):
#                 res.append(matrix[top][i])
#             top += 1
#             for i in range(top, bottom):
#                 res.append(matrix[i][right-1])
#             right -= 1
#             if not (left < right and top < bottom):
#                 break
#             for i in range(right-1, left-1, -1):
#                 res.append(matrix[bottom-1][i])
#             bottom -= 1
#             for i in range(bottom-1, top-1, -1):
#                 res.append(matrix[i][left])
#             left += 1
#         return res

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        left, top = 0, 0
        right, bottom = len(matrix[0]), len(matrix)
        res = []
        while left < right and top < bottom:
            for i in range(left, right):
                res.append(matrix[top][i])
            top += 1
            if top >= bottom:
                break
            for i in range(top, bottom):
                res.append(matrix[i][right-1])
            right -= 1
            if left >= right:
                break
            for i in range(right-1, left-1, -1):
                res.append(matrix[bottom-1][i])
            bottom -= 1
            if top >= bottom:
                break
            for i in range(bottom-1, top-1, -1):
                res.append(matrix[i][left])
            left += 1
        return res
```

### 70. [Leetcode 73](https://leetcode.com/problems/set-matrix-zeroes/description/) : Set Matrix Zeroes

<pre>
Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.

You must do it in place.

 

Example 1:


Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]
Example 2:


Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
 

Constraints:

m == matrix.length
n == matrix[0].length
1 <= m, n <= 200
-231 <= matrix[i][j] <= 231 - 1
 

Follow up:

A straightforward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?
</pre>

Solution:

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        row_zero = False
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    matrix[0][j] = 0
                    if i > 0:
                        matrix[i][0] = 0
                    else:
                        row_zero = True

        for i in range(1, len(matrix)):
            for j in range(1, len(matrix[0])):
                if matrix[0][j] == 0 or matrix[i][0] == 0:
                    matrix[i][j] = 0
        
        if matrix[0][0] == 0:
            for i in range(len(matrix)):
                matrix[i][0] = 0
                
        if row_zero == True:
            for j in range(len(matrix[0])):
                matrix[0][j] = 0
        return matrix

# class Solution:
#     def setZeroes(self, matrix: List[List[int]]) -> None:
#         """
#         Do not return anything, modify matrix in-place instead.
#         """
#         row_array = set()
#         col_array = set()
#         for i in range(len(matrix)):
#             for j in range(len(matrix[0])):
#                 if matrix[i][j] == 0:
#                     row_array.add(i)
#                     col_array.add(j)
#         for i in row_array:
#             for j in range(len(matrix[0])):
#                 matrix[i][j] = 0
#         for j in col_array:
#             for i in range(len(matrix)):
#                 matrix[i][j] = 0
#         return matrix
```

## Bit Manipulation

### 71. [Leetcode 191](https://leetcode.com/problems/number-of-1-bits/description/) : Number of 1 Bits

<pre>
Given a positive integer n, write a function that returns the number of 
set bits
 in its binary representation (also known as the Hamming weight).

 

Example 1:

Input: n = 11

Output: 3

Explanation:

The input binary string 1011 has a total of three set bits.

Example 2:

Input: n = 128

Output: 1

Explanation:

The input binary string 10000000 has a total of one set bit.

Example 3:

Input: n = 2147483645

Output: 30

Explanation:

The input binary string 1111111111111111111111111111101 has a total of thirty set bits.

 

Constraints:

1 <= n <= 231 - 1
 

Follow up: If this function is called many times, how would you optimize it?
</pre>

Solution:

```python
# class Solution:
#     def hammingWeight(self, n: int) -> int:
#         res = 0
#         while n > 0:
#             if n %2 > 0:
#                 res += 1
#             n = n // 2
#         return res

# class Solution:
#     def hammingWeight(self, n: int) -> int:
#         res = 0
#         while n > 0:
#             res += n % 2
#             n = n >> 1 # Bit right shift one meaning divide by 2
#         return res

class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            n = n & (n - 1)
            res += 1
        return res
```

### 72. [Leetcode 338](https://leetcode.com/problems/counting-bits/description/) : Counting Bits

<pre>
Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.

 

Example 1:

Input: n = 2
Output: [0,1,1]
Explanation:
0 --> 0
1 --> 1
2 --> 10
Example 2:

Input: n = 5
Output: [0,1,1,2,1,2]
Explanation:
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101
 

Constraints:

0 <= n <= 105
 

Follow up:

It is very easy to come up with a solution with a runtime of O(n log n). Can you do it in linear time O(n) and possibly in a single pass?
Can you do it without using any built-in function (i.e., like __builtin_popcount in C++)?
</pre>

Solution:

```python
# class Solution:
#     def countBits(self, n: int) -> List[int]:
#         res = [0] * (n+1)
#         for i in range(1, n+1):
#             num = i
#             while num > 0:
#                 if res[num] > 0:
#                     res[i] += res[num]
#                     break
#                 res[i] += (num % 2)
#                 num = num // 2
#         return res


class Solution:
    def countBits(self, n: int) -> List[int]:
        res = [0] * (n+1)
        offset = 1
        for i in range(1, n+1):
            if offset * 2 == i:
                offset = i
            res[i] = 1 + res[i - offset] 
        return res
```

### 73. [Leetcode 190](https://leetcode.com/problems/reverse-bits/description/) : Reverse Bits

<pre>
Reverse bits of a given 32 bits unsigned integer.

Note:

Note that in some languages, such as Java, there is no unsigned integer type. In this case, both input and output will be given as a signed integer type. They should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 2 above, the input represents the signed integer -3 and the output represents the signed integer -1073741825.
 

Example 1:

Input: n = 00000010100101000001111010011100
Output:    964176192 (00111001011110000010100101000000)
Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, so return 964176192 which its binary representation is 00111001011110000010100101000000.
Example 2:

Input: n = 11111111111111111111111111111101
Output:   3221225471 (10111111111111111111111111111111)
Explanation: The input binary string 11111111111111111111111111111101 represents the unsigned integer 4294967293, so return 3221225471 which its binary representation is 10111111111111111111111111111111.
 

Constraints:

The input must be a binary string of length 32
 

Follow up: If this function is called many times, how would you optimize it?
</pre>

Solution:

```python
class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for i in range(32):
            bit = (n >> i) & 1
            res = res | (bit << 31 - i)
        return res

# class Solution:
#     def reverseBits(self, n: int) -> int:
#         _binary = [0] * 32
#         i = 0
#         while n:
#             _binary[i] = n % 2
#             n = n >> 1
#             i += 1
#         i = 0
#         res = 0
#         while i < 32:
#             res += math.pow(2, i) * _binary[31-i]
#             i += 1
#         return int(res)
```


### 74. [Leetcode 268](https://leetcode.com/problems/missing-number/description/) : Missing Number

<pre>
Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.

 

Example 1:

Input: nums = [3,0,1]
Output: 2
Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 2 is the missing number in the range since it does not appear in nums.
Example 2:

Input: nums = [0,1]
Output: 2
Explanation: n = 2 since there are 2 numbers, so all numbers are in the range [0,2]. 2 is the missing number in the range since it does not appear in nums.
Example 3:

Input: nums = [9,6,4,2,3,5,7,0,1]
Output: 8
Explanation: n = 9 since there are 9 numbers, so all numbers are in the range [0,9]. 8 is the missing number in the range since it does not appear in nums.
 

Constraints:

n == nums.length
1 <= n <= 104
0 <= nums[i] <= n
All the numbers of nums are unique.
 

Follow up: Could you implement a solution using only O(1) extra space complexity and O(n) runtime complexity?
</pre>

Solution:

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        expected_sum = (n * (n+1)) // 2
        _sum = 0
        for num in nums:
            _sum += num
        return expected_sum - _sum
```

### 75. [Leetcode 371](https://leetcode.com/problems/sum-of-two-integers/description/) : Sum of Two Integers

<pre>
Given two integers a and b, return the sum of the two integers without using the operators + and -.

 

Example 1:

Input: a = 1, b = 2
Output: 3
Example 2:

Input: a = 2, b = 3
Output: 5
 

Constraints:

-1000 <= a, b <= 1000
</pre>

Solution:

```python
# Actual solution

# class Solution:
#     def getSum(self, a: int, b: int) -> int:
#         _sum = a ^ b
#         carry = (a & b) << 1
#         while carry != 0:
#             a = _sum
#             b = carry
#             _sum = a ^ b
#             carry = (a & b) << 1
#         return _sum

# Solution with masking due to Integer size issue in Python

class Solution:
    def getSum(self, a: int, b: int) -> int:
        mask = 0xFFFFFFFF
        max_int = 0x7FFFFFFF
        _sum = (a ^ b) & mask
        carry = ((a & b) << 1) & mask
        while carry != 0:
            a = _sum
            b = carry
            _sum = (a ^ b) & mask
            carry = ((a & b) << 1) & mask
        return _sum if _sum <= max_int else ~(_sum ^ mask)

```
