#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "opencv2/world.hpp"
#include <QLabel>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();



private slots:
    void on_actionUpload_triggered();

    void on_harrisBtn_clicked();

    void on_harrisBtn_2_clicked();

    void on_set_seedsBtn_clicked();

    void on_submitBtn_clicked();

    void on_comboBox_currentTextChanged(const QString &arg1);

    void on_submitBtn_2_clicked();

    void on_kmeansBtn_clicked();

    void on_horizontalSlider_sliderMoved(int position);

    void on_horizontalSlider_actionTriggered(int action);

private:
    Ui::MainWindow *ui;
};
#endif // MAINWINDOW_H
