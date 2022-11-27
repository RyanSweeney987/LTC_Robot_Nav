# dictionary format
#		"command A": ["sub command A", "sub command B"],
#		"command B": ["sub command C"],
#		"command C": [] <------- No sub commands, executes on base command
#
# 	{
# 		"first_word_a": ["second_word_a", "second_word_b"],
#		"first_word_b": ["second_word_c"]				
# 	}

# Alt word format
#	["full phrase A", "full phrase B"]
class CommandProcessor:
	""" This class processes words and returns the appropriate response """

	def __init__(self, wordDict, phraseDict):
		self.wordDict = wordDict
		self.phraseDict = phraseDict
		self.currentWord = ""
		self.previousWord = ""
		self.hasRecievedCommand = False
		self.hasReceivedSubCommand = False

	def processWord(self, newWord):
		self.previousWord = self.currentWord
		self.currentWord = newWord

		result = (False, "", "")

		# Reset if both a command and sub command has been processed to allow for new commands
		if self.hasRecievedCommand and self.hasReceivedSubCommand:
			self.hasRecievedCommand = False
			self.hasReceivedSubCommand = False
	
		# If the word is a valid key, continue
		if not self.hasRecievedCommand:
			if newWord in self.wordDict.keys():
				self.hasRecievedCommand = True
				# If the word has sub commands, wait for it otherwise return the word as final output
				if len(self.wordDict.get(self.currentWord)) > 0:
					result = (True, self.currentWord, "")
				else:
					self.hasReceivedSubCommand = True
					result = (True, self.currentWord, self.currentWord)
		elif self.hasRecievedCommand and not self.hasReceivedSubCommand:
			# Only continue if a command has already been processed and continue again if a valid sub command exists under the command
			if newWord in self.wordDict.get(self.previousWord):
				self.hasReceivedSubCommand = True
				result = (True, self.previousWord, self.currentWord)
		
		# Reset if the word isn't a valid command or sub command
		if not result[0]:
			self.hasRecievedCommand = False
			self.hasReceivedSubCommand = False

		return result

	def processWord2(self, newPhrase):
		self.previousWord = self.currentWord
		self.currentWord = newPhrase

		result = (False, self.phraseDict.get(""))

		if self.currentWord in self.phraseDict.keys():
			result = (True, self.phraseDict.get(self.currentWord))

		return result
