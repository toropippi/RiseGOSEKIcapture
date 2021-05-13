# RiseGOSEKIcapture
 モンハンライズの護石を画像認識してテキスト化します。  
 護石が多すぎて管理しきれない、護石テーブルを自分で確かめたいけど手動でスキル内容を打ち込むのは面倒・・・という方に需要があるかと思い作りました。  
 #モンハンライズ #MHR #MHRise #護石 #護石認識 #文字認識 #画像認識 #自動化 #手動 #テキスト化 #OpenCL
## 使い方
　
### GOSEKIcapture2.exeの起動
　GOSEKIcapture2.exeを起動します。  
 <img width="711" alt="0" src="https://user-images.githubusercontent.com/44022497/116805810-ce870800-ab63-11eb-84f9-0acf59be8de2.png">  
 すると白いウィンドウの他に緑色の枠がでます。これもウィンドウなのでマウスドラッグで移動できます。  
 ![mouse](https://user-images.githubusercontent.com/44022497/116805822-e65e8c00-ab63-11eb-8465-e4d488d7f9c8.png)  

### キャプチャシーンに合わせる
 緑の枠をプレイ動画などのキャプチャシーンに合わせます。  
 このときプレイ画面は1920×1080でないと枠に合いません。Nintendo Switchの録画ボタンで録画した動画の場合1280×720なので、手動で動画サイズを引き伸ばすなりして下さい。  
 なるべく1920×1080に近づけて、枠も護石内容の文字がおさまるようにしてください。ここがアバウトだと認識精度に影響します。  
 ![1](https://user-images.githubusercontent.com/44022497/116805853-2b82be00-ab64-11eb-9223-5188be1233b8.jpg)

### スライドバーを移動させてカットオフ値を決める
 スライドバーを左右に移動させて適切なカットオフ値にして下さい。  
 下の画像のようにカットオフ値がMaxの場合、認識性能は上がりますが計算が重くなり、キャプチャフレームレートに影響します。  
 <img alt="3" src="https://user-images.githubusercontent.com/44022497/118156548-73eb8700-b454-11eb-9f4c-af851bbd0200.jpg">  
 認識しないよりはマシなので**基本カットオフ値Max**で良いと思います。護石のスキル候補が下に表示されます。  
 スキル名と、そのスキルの可能性が数値で表されます。数値は低いほど可能性が高いという意味です。  
 例えば「水属性攻撃強化」と「氷属性攻撃強化」は文字が非常によく似ているのでどちらも候補に出現することがありますが、より可能性の高いほうが採用されているのが数値を見て分かったりするので、眺めているだけでも楽しいです(人によります)。

### mesbox(テキスト画面)のOnOff
 「←」と書いてあるボタンを押すとmesbox(テキスト画面)が消えます。  
 テキスト内容に変更があったら常に更新をかけるので地味にフレームレートに影響することがあるので一応つけました。  
 もちろんテキスト画面Offにしても護石認識は機能します。  
 <img width="221" alt="4" src="https://user-images.githubusercontent.com/44022497/116805874-51a85e00-ab64-11eb-868a-b722e37ae34d.png">  

### コピペ
 <img alt="5" src="https://user-images.githubusercontent.com/44022497/118156669-9a112700-b454-11eb-940f-147342dc66f4.jpg">  
 <img alt="6" src="https://user-images.githubusercontent.com/44022497/118156776-b614c880-b454-11eb-9aeb-e280c720457b.jpg">  

テキスト出力は , 区切りで  
「スキル名1,Lv,スキル名2,Lv,スロA,スロB,スロC」  
となっています。  

## 注意事項
  
### 文字化け(ver1でもver2でも)
 約1000～1500個以上の護石をリストアップすると、文字数が32768(2の15乗)を超えて正しく表示されなくなるバグがあります。  
 ![mojibake](https://user-images.githubusercontent.com/44022497/116907491-9de6c180-ac7c-11eb-9095-3a4ab1ddae65.png)  
 これは表示の問題であり、ソフトの内部では正しい文字列で記憶されています。**Escキー**を押してソフトを終了することにより正しい文字列が「テキスト出力.txt」という名前で保存されます。  
 
### UIがはみ出てる
 ![juhuku](https://user-images.githubusercontent.com/44022497/116908342-bc00f180-ac7d-11eb-8df5-df91b7788260.png)  
 これは正しく表示されているときですが、「重複を数える」というチェックボックスが正しく表示されない場合があります。  
 その場合ディスプレイの表示スケールの設定を100%に変更して下さい。  
 ![kakusyuku](https://user-images.githubusercontent.com/44022497/116908436-ddfa7400-ac7d-11eb-8a3b-601a162182af.jpg)  
 これが150%とか200%だと微妙にUIが崩れてしまうことはありますが、ソフトの機能自体に不具合が起こることはありません。  
  
### 装飾品がついている場合
 装飾品のせいでスロ3がスロ1に誤認識されてしまうことがあります。  
 そのうち修正します。  

### OpenCLデバイスがない
 GOSEKIcapture ver2からOpenMPではなくOpenCLを使って計算をしています。  
 お使いのPCが古い場合「HSPCL64.dllがありません」といったようなエラーが出ると思います。  
 ~~ごみおまの前にそのごみPCを捨てましょう~~  
 ver1を別フォルダに残しているのでそちらを起動してみて下さい。  

### 計算に使うGPU,CPUを手動で指定したい
 HCLGetDeviceInfo.exeを起動するとOpenCLデバイスが列挙されます。  
 <img alt="7" src="https://user-images.githubusercontent.com/44022497/118158359-a0a09e00-b456-11eb-9e47-34c5f08bf0aa.jpg">  
 使いたいGPU,CPUのデバイスidをGOSEKIcapture2Setting.txtの2行目に記載することで、任意のデバイスで計算を行うことができます。  
 GOSEKIcapture2.exeを起動すると、計算時間がタイトルバーに表示されます。単位はマイクロセカンド(us)です。  
 OpenCLデバイスがない場合、HCLGetDeviceInfo.exeを起動してもデバイスが列挙されませんので、確認用に使いこともできます。  

## スキル文字認識の原理
 テンプレートマッチングみたいなことをやってます。  
 そのうちディープラーニングで認識精度と速度の両立を図りたい・・・  
 
## 更新履歴
2021/5/14 ver 2.0 テンプレートマッチング計算部分をOpenCLに移植して高速化  
2021/5/3 ver 1.11 重複を数えない設定を追加  
2021/5/3 ver 1.10 達人芸を対応。出力文字数が32767を超えるとバグるのを修正  
2021/5/2 ver 1.0 公開  
  
---------------------------------------------------------------  
# GOSEKIcapture2_CUI  
 他のソフトから起動されることを想定されたソフトです。  
 コマンドライン引数に画像ファイル名を指定することで、スキル認識を行い結果をテキストで出力します。  
## 対応形式
 画像ファイルは  
 ・png
 ・jpg
 ・bmp
 形式をサポートしています。  
 解像度は1920x1080や1280x720の16:9画像が対応しています。  
## 実行
 実行形式ですが、例えばこのような形で実行することができます。  
 ![readme0](https://user-images.githubusercontent.com/44022497/118168339-b5832e80-b462-11eb-98bc-c99341365509.png)  
## 結果出力
 結果出力は下記のようになります。  
 ![readme2](https://user-images.githubusercontent.com/44022497/118168392-c5027780-b462-11eb-8c7d-96afbd2ecd99.png)  
 実行ファイルと同じ階層に出力されます。  
## デバッグ出力
 ↓これはデバッグ機能みたいなものです。  
 ![readme1](https://user-images.githubusercontent.com/44022497/118168390-c469e100-b462-11eb-90ea-1b06ffe8b265.png)  
 実際にキャプチャされた部分が切り出されて出力されます。  
 またソフト内で引数がどのような文字列で処理されたかも見ることができます。  
 
## 読み取り座標指定
 GOSEKIcapture2Setting.txtの座標数値を変えることで任意の座標からの読み取りが可能になります。  
 デフォルトでは  
![mhr](https://user-images.githubusercontent.com/44022497/118169366-d1d39b00-b463-11eb-8558-dd0332087345.jpg)  
 このシーンの読み取りを行う設定になっています。
