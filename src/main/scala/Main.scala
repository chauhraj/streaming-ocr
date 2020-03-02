import java.awt.image.BufferedImage
import java.io.File

import akka.NotUsed
import akka.actor.ActorSystem
import akka.http.scaladsl.common.{EntityStreamingSupport, JsonEntityStreamingSupport}
import akka.http.scaladsl.marshallers.sprayjson.SprayJsonSupport
import akka.stream.SystemMaterializer
import akka.stream.scaladsl.{Flow, Sink, Source}
import com.joestelmach.natty.Parser
import com.recognition.software.jdeskew.{ImageDeskew, ImageUtil}
import javax.imageio.ImageIO
import net.sourceforge.tess4j.Tesseract
import net.sourceforge.tess4j.util.ImageHelper
import opennlp.tools.namefind.{NameFinderME, TokenNameFinderModel}
import opennlp.tools.sentdetect.{SentenceDetectorME, SentenceModel}
import opennlp.tools.tokenize.{TokenizerME, TokenizerModel}
import opennlp.tools.util.Span
import org.bytedeco.javacpp.indexer.UByteRawIndexer
import org.bytedeco.javacv.Java2DFrameUtils
import org.bytedeco.opencv.opencv_core.Mat
import spray.json.{DefaultJsonProtocol, RootJsonFormat}

import scala.concurrent.{ExecutionContextExecutor, Future}

object Main extends OCR with Spell with NLP with Natty {
  implicit val system: ActorSystem = ActorSystem("ocr")
  implicit val executor: ExecutionContextExecutor = system.dispatcher
  implicit val materializer = SystemMaterializer(system)
  implicit val jsonStreamingSupport: JsonEntityStreamingSupport = EntityStreamingSupport.json()

  def imageDeSkew(skewThreshold:Double = 0.050) = Flow[BufferedImage].map(bi => {
    val deSkew = new ImageDeskew(bi)
    val imageSkewAngle = deSkew.getSkewAngle

    if (imageSkewAngle > skewThreshold || imageSkewAngle < -skewThreshold) {
      ImageUtil.rotate(bi, -imageSkewAngle, bi.getWidth() / 2, bi.getHeight() / 2)
    } else {
      bi
    }
  })

  def imageToBinaryImage = Flow[BufferedImage].map(img => {
    val bin = ImageHelper.convertImageToBinary(img)
    ImageUtil.rotate(bin, -90, 0, 0)
  })

  def bufferedImageToMat = Flow[BufferedImage].map(bi => {
    import org.bytedeco.opencv.global.opencv_core.CV_8UC
    val mat = new Mat(bi.getHeight, bi.getWidth, CV_8UC(3))
    val indexer:UByteRawIndexer = mat.createIndexer()
    for (y <- 0 until bi.getHeight()) {
      for (x <- 0 until bi.getWidth()) {
        val rgb = bi.getRGB(x, y)
        indexer.put(y, x, 0, (rgb >> 0) & 0xFF)
        indexer.put(y, x, 1, (rgb >> 8) & 0xFF)
        indexer.put(y, x, 2, (rgb >> 16) & 0xFF)
      }
    }
    indexer.release()
    mat
  })

  def matToBufferedImage = Flow[Mat].map(mat => {
    Java2DFrameUtils.toBufferedImage(mat)
  })

  def enhanceMat = Flow[Mat].map(mat => {
    val src = mat.clone()
    import org.bytedeco.opencv.global.{opencv_photo => Photo}
    Photo.fastNlMeansDenoising(mat, src, 40, 7, 21)
    val dst = src.clone()
    Photo.detailEnhance(src, dst)
    dst
  })

  private def extractPersons = Flow[OcrSuggestions].map(ocr => {
    val tokens = tokenizer.tokenize(ocr.ocr)
    val spans:Array[Span] = personFinderME.find(tokens)
    val persons = spans.toList.map(span => tokens(span.getStart()))
    OcrSuggestionsPersons(ocr.ocr, ocr.suggestions, persons)
  })

  private def extractDates: Flow[OcrSuggestionsPersons, OcrSuggestionsPersonsDates, NotUsed] = Flow[OcrSuggestionsPersons].map(ocr => {
    val sentences = sentenceDetector.sentDetect(ocr.ocr.replaceAll("\n", " ")).toList
    import scala.jdk.CollectionConverters._
    val dates = sentences.map(sentence => parser.parse(sentence))
      .flatMap(dateGroups => dateGroups.asScala.toList)
      .map(dateGroup => (dateGroup.getDates().asScala.toList.map(_.toString()), dateGroup.getText()))

    OcrSuggestionsPersonsDates(ocr.ocr, ocr.suggestions, ocr.persons, dates)
  })

  private def spellCheck = Flow[String].map(ocr => {
    println(ocr)
    import scala.jdk.CollectionConverters._
    val words: Set[String] = ocr.replaceAll("-\n", "").replaceAll("\n", " ").replaceAll("-"," ").split("\\s+")
      .map(_.replaceAll(
      "[^a-zA-Z'’\\d\\s]", "") // Remove most punctuation
      .trim)
      .filter(!_.isEmpty).toSet
      println(words)
    val misspelled = words.filter(word => !speller.isCorrect(word))
    val suggestions: Set[Map[String, List[String]]] = misspelled.map(mis => {
      Map(mis -> speller.suggest(mis).asScala.toList)
    })
    OcrSuggestions(ocr, suggestions)
  })

  private def imageOcr = Flow[BufferedImage].map{ bi =>
    val content = tesseract.doOCR(bi)
    println("OCR Content:")
    println(s"$content")
    content
  }

  private def imageSink(path:String, format:String = "png") = Sink.foreachAsync[BufferedImage](4)(bi => {
    Future( ImageIO.write(bi, format, new File(path)) )
  })

  private val imageEnhance = bufferedImageToMat.via(enhanceMat).via(matToBufferedImage)

  private val imagePreProcessFlow =
    imageToBinaryImage.alsoTo(imageSink("binary.png"))
    .via(imageEnhance).alsoTo(imageSink("enhanced.png"))
    .via(imageDeSkew()).alsoTo(imageSink("deskew.png"))

  val ocrFlow = imageOcr //.via(spellCheck).via(extractPersons).via(extractDates)

  def main(args: Array[String]): Unit = {
    Source.single(new File(s"${sys.props("user.dir")}/data/input"))
      .mapConcat(_.listFiles().toList)
      .mapConcat(_.listFiles().toList)
      .alsoTo(Sink.foreach(file => println(s"Processing image:${file.getAbsolutePath}")))
      .map(ImageIO.read)
      .via(imagePreProcessFlow)
      .via(ocrFlow)
      .runWith(Sink.ignore)
  }
  import scala.concurrent.duration._
  system.scheduler.scheduleOnce(30.seconds) {
    system.terminate().foreach(_ => println("Exiting..."))
  }

}

case class OcrSuggestions(ocr:String, suggestions: Set[Map[String, List[String]]])
case class OcrSuggestionsPersons(ocr:String, suggestions: Set[Map[String, List[String]]], persons: List[String])
case class OcrSuggestionsPersonsDates(ocr:String, suggestions: Set[Map[String, List[String]]], persons: List[String], dates: List[(List[String], String)])

trait OCR {
  lazy val tesseract: Tesseract = {
    val instance = new Tesseract
    instance.setDatapath("/usr/share/tesseract-ocr/4.00/tessdata/")
    instance.setTessVariable("user_defined_dpi", "300")
    instance
  }
}

trait Spell {
  import com.atlascopco.hunspell.Hunspell
  lazy val speller = new Hunspell("src/main/resources/en_US.dic", "src/main/resources/en_US.aff")
}

trait NLP {
  lazy val tokenModel = new TokenizerModel(getClass.getResourceAsStream("/en-token.bin"))
  lazy val tokenizer = new TokenizerME(tokenModel);

  lazy val sentenceModel = new SentenceModel(getClass.getResourceAsStream("/en-sent.bin"))
  lazy val sentenceDetector = new SentenceDetectorME(sentenceModel);

  lazy val personModel = new TokenNameFinderModel(getClass.getResourceAsStream("/en-ner-person.bin"))
  lazy val personFinderME = new NameFinderME(personModel);
}

trait Natty {
  lazy val parser = new Parser()
}

object MyJsonProtocol
  extends SprayJsonSupport
    with DefaultJsonProtocol {
  implicit val ocrFormat: RootJsonFormat[OcrSuggestions] = jsonFormat2(OcrSuggestions.apply)
  implicit val ocr2Format: RootJsonFormat[OcrSuggestionsPersons] = jsonFormat3(OcrSuggestionsPersons.apply)
  implicit val ocr3Format: RootJsonFormat[OcrSuggestionsPersonsDates] = jsonFormat4(OcrSuggestionsPersonsDates.apply)
}
