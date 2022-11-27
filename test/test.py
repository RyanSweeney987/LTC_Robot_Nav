from inaoqi import ALMemoryProxy
from naoqi import ALProxy, ALModule
import time
import sys

check = 0

# create python module
class myModule(ALModule):
	""" python class myModule test auto documentation: comment needed to create a new python module """

	def pythondatachanged(self, key, value, message):
		""" callback when data change """
		print("datachanged", key, " ", value, " ", message)
		global check
		check = 1

	def __pythonPrivateMethod(self, param1, param2, param3):
		""" Mandatory docstring. Comment needed to create a bound method """
		global check

def main():
	ipAddress = "169.254.162.107"
	ipAddress = "127.0.0.1"
	port = 9559

	memory = ALProxy("ALMemory", ipAddress, port)
	speechRecog = ALProxy("ALSpeechRecognition", ipAddress, port)
	nav = ALProxy("ALNavigation", ipAddress, port)
	motion = ALProxy("ALMotion", ipAddress, port)

	speechRecog.setLanguage("English")
	speechRecog.pause(True)
	speechRecog.subscribe("WordRecognized")
	
	try:
		speechRecog.setVocabulary(["move", "stop"], False)
	except Exception as e:
		print("Error updating vocabulary: ", e)
	finally:
		speechRecog.pause(False)
	
	#print(speechRecog.getMethodHelp("setWordListAsVocabulary"))
	#print(speechRecog.getMethodHelp("setVocabulary"))
	#print(speechRecog.getMethodHelp("createContext"))
	print(nav.getMethodList())
	#print(memory.getMethodHelp("insertData"), "\n")
	print(nav.getMethodHelp("navigateTo"), "\n")
	#print(memory.getDescriptionList(["WordRecognized"]), "\n")
	#print(memory.getDataList(), "\n")
	#print(memory.getMethodHelp("subscribeToEvent"))
	t = memory.subscribeToEvent("WordRecognized", "myModule", "pythondatachanged")

	#nav.subscribe("Navigation/AvoidanceNavigator/Status")

	currentWord = []
	prevWord = []
	running = True
	isMoving = False
	memory.insertData("WordRecognized", "")

	while running:
		currentWord = list(memory.getData("WordRecognized"))

		print(currentWord, "\n", "IsMoving: ", isMoving)
		sys.stdout.write("\033[F")

		if currentWord != [] and prevWord != currentWord:		
			if currentWord[1] > 0.4:
				if currentWord[0] == "move" and not isMoving:
					isMoving = True
					#nav.moveAlong(["Composed",["Holonomic", ["Line", [1.0, 0.0]], 0.0, 1.0]])
					#nav.moveTo(0.5, 0)
					#nav.navigateTo(1, 0)
					motion.move(0.5, 0.0, 0.0)
					motion.moveTo(0.5, 0.0, 0.0)
					motion.moveTowards(0.5, 0.0, 0.0)

				if currentWord[0] == "stop":
					running = False
					if isMoving:
						nav.stopNavigateTo()

			prevWord = currentWord
			currentWord = []

		#memory.insertData("WordRecognized", "")

		time.sleep(1 / 30)

if __name__ == "__main__":
	main()