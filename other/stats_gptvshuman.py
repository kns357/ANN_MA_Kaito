from numpy import *
from math import *
import torch

t_human = """Gerade fühlt sich alles ziemlich garstig an. Während viele in unserem Umfeld mit Grippe im Bett liegen, baut der neue US-Präsident im Eiltempo den Staatsapparat um (hier dazu ein sehr lesenswerter Artikel in der Republik). Und während ein Teil des Festival-Teams noch in der Winterpause ruht, suchen wir bereits nach den ersten literarischen Frühlingszeichen – vor allem aber nach neuen Texten. 

Zusammen mit der Basler Kantonalbank lancieren wir den Schreibwettbewerb Basel Schreibt. Der Wettbewerb richtet sich an Schreibende aller Altersgruppen – was zählt, sind frische und kreative Ideen. Gemeinsam in Basel lautet das vorgegebene Thema. Es dient als Ausgangslage für den Wettbewerbsbeitrag und soll auf originelle sowie spannende Weise literarisch umgesetzt werden. Alle Gattungen sind erlaubt: Ob Prosa, Lyrik, Comic oder Essay – wir freuen uns auf eure Einsendungen! 

Einsendeschluss ist der 1. März 2025. Zu gewinnen gibt es Preise bis zu CHF 5'000.–. 

Die prämierten Texte werden im Rahmen einer feierlichen Preisverleihung am BuchBasel 2025 vorgestellt. Mehr Informationen findet ihr auf unserer Webseite. 

Wir freuen uns auf eure Beiträge!

Viele Grüsse,
das Festival-Team
"""

t_gpt = """
Der Schreibwettbewerb „Basel Schreibt“ geht in eine neue Runde – dieses Mal gemeinsam mit der Basler Kantonalbank. Ob jung oder alt, Vielschreiberin oder Gelegenheitsautor: Eingeladen sind alle, die Lust haben, Worte in Bewegung zu setzen.

Das Thema 2025 lautet: „Gemeinsam in Basel“.
Was bedeutet das – gemeinsam? In einer Stadt voller Sprachen, Geschichten, Wege, Ideen? Welche Bilder tauchen auf, wenn du an Zusammenleben in Basel denkst? Vielleicht ist es ein Moment im Tram, ein Streit im Treppenhaus, ein Tag am Rhein, ein leises Gespräch oder eine große Geste. Was auch immer du daraus machst – wir wollen es lesen.

Egal, ob du lieber erzählst, dichtest, reflektierst oder zeichnest: Alle literarischen Formen sind erlaubt. Prosa, Lyrik, Essay oder Comic – Hauptsache, es ist deine Stimme.

Einsendeschluss ist der 1. März 2025.
Die besten Texte werden nicht nur mit Preisgeldern von bis zu 5.000 CHF belohnt, sondern auch auf der BuchBasel 2025 vor Publikum vorgestellt.

Alle Details zum Wettbewerb und zur Einreichung findest du online auf der offiziellen Webseite.

Wenn dir etwas einfällt – schreib’s auf. Wenn dir noch nichts einfällt – vielleicht ist jetzt der Moment.
"""

h = 0
g = 0

for word in list(t_human):
    h += len(word)

for word in list(t_gpt):
    g += len(word)

print(len(list(t_human)), len(list(t_gpt)))